"""Judge evidence sufficiency and outcome leakage."""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .llm import AnthropicLLMClient
from prompts import PROMPTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "claude-opus-4-6"

JUDGE_PROMPT_PATH = PROMPTS_DIR / "evidence_check" / "judge.md"

VALID_VERDICTS: tuple[str, ...] = ("SUFFICIENT", "PARTIAL", "INSUFFICIENT")
VALID_INFORMATIVENESS: tuple[int, ...] = (0, 1, 2, 3)

MAX_KEY_HITS = 5

DEFAULT_MAX_HITS_IN_PROMPT = 15


@dataclass
class JudgeResult:
    """One question's informativeness verdict + provenance."""

    event_id: str
    slug: str
    title: str
    verdict: str = ""
    informativeness: int = -1
    leakage_detected: bool = False
    leakage_reason: str | None = None
    confident_lean: str | None = None
    key_hits: list[str] = field(default_factory=list)
    reasoning: str = ""
    model_id: str = ""
    n_hits_seen: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "slug": self.slug,
            "title": self.title,
            "verdict": self.verdict,
            "informativeness": self.informativeness,
            "leakage_detected": self.leakage_detected,
            "leakage_reason": self.leakage_reason,
            "confident_lean": self.confident_lean,
            "key_hits": self.key_hits,
            "reasoning": self.reasoning,
            "model_id": self.model_id,
            "n_hits_seen": self.n_hits_seen,
            "error": self.error,
        }


def _load_system_prompt() -> str:
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


def _format_question_block(q: dict) -> str:
    """Render question facts without the realized outcome or market probabilities."""
    title = q.get("title") or q.get("slug") or ""
    body = (q.get("body") or "").strip()
    kind = q.get("belief_kind") or "binary"
    # Prefer recorded settlement over the creator-set deadline.
    res_date = ((q.get("close_date") or q.get("resolution_date")
                 or q.get("end_date") or "")[:10])

    if kind == "binary":
        outcomes = "Outcomes: YES / NO."
    else:
        legs = "\n".join(
            f"  - {m.get('label')!r}: {m.get('question') or '(no per-leg question)'}"
            for m in (q.get("markets") or [])
        )
        outcomes = (
            f"Outcomes ({len(q.get('markets') or [])}, exactly one resolves YES). "
            f"Each leg's resolution criterion is the question after its label:\n"
            f"{legs}"
        )

    parts = [
        f"Question: {title}",
        f"Resolves: {res_date}",
        outcomes,
    ]
    if body:
        parts.append(f"\nResolution criteria / context:\n{body}")
    return "\n".join(parts)


def _format_hit_block(hit: dict, rank: int) -> str:
    """Render article text, citation ID, retrieval scores, and publication date."""
    title = (hit.get("title") or "(no title)").strip()
    url = (hit.get("url") or "(no url)").strip()
    pub = (hit.get("published_date") or "(no date)").strip()
    aid = hit.get("article_id") or ""
    rrf = hit.get("rrf_score")
    nq = hit.get("n_query_hits")
    text = (hit.get("text") or "").strip()

    meta = []
    if rrf is not None:
        meta.append(f"rrf={rrf:.4f}")
    if nq is not None:
        meta.append(f"queries_hit={nq}")
    if pub:
        meta.append(f"published={pub}")
    meta_line = "  [" + " | ".join(meta) + "]" if meta else ""

    return (
        f"## Hit {rank}\n"
        f"article_id: {aid}\n"
        f"title: {title}\n"
        f"url: {url}\n"
        f"{meta_line}\n\n"
        f"{text}"
    )


def _hit_age_stats(
    hits: list[dict], cutoff: str | None,
) -> dict[str, Any]:
    """Summarize article ages relative to the retrieval cutoff, ignoring invalid dates."""
    from datetime import date

    if not cutoff:
        return {}
    try:
        c = date.fromisoformat(cutoff[:10])
    except ValueError:
        return {}
    ages: list[int] = []
    for h in hits:
        pd = (h.get("published_date") or "")[:10]
        if not pd:
            continue
        try:
            ages.append((c - date.fromisoformat(pd)).days)
        except ValueError:
            continue
    if not ages:
        return {"n_with_date": 0}
    ages.sort()
    median = ages[len(ages) // 2]
    return {
        "n_with_date": len(ages),
        "freshest_hit_age_days": min(ages),
        "median_hit_age_days": median,
        "hits_within_30d": sum(1 for a in ages if a <= 30),
        "hits_within_7d": sum(1 for a in ages if a <= 7),
    }


def _format_user_prompt(
    q: dict, retrieved: dict, *, max_hits: int,
) -> tuple[str, int]:
    """Render the question, retrieval window, and ranked hits; return (prompt, count)."""
    hits = (retrieved.get("hits") or [])[:max_hits]
    window = retrieved.get("window") or [None, None]
    start = window[0] if window else None
    cutoff = window[1] if len(window) >= 2 else None

    parts = [_format_question_block(q)]
    parts.append("")
    parts.append(
        f"Retrieval window: [{start or '?'}, {cutoff or '?'}]  "
        f"(only articles published in this range were retrieved; if any hit "
        f"falls outside, that's a leakage signal)"
    )
    parts.append(f"n_queries used in retrieval: {retrieved.get('n_queries')}")
    stats = retrieved.get("stats") or {}
    if stats:
        parts.append(
            f"retrieval stats: dense_hits={stats.get('dense_hits')} "
            f"bm25_hits={stats.get('bm25_hits')} "
            f"fused_unique={stats.get('fused_unique_docs')} "
            f"queries_zero_hits={stats.get('queries_zero_hits')}"
        )
    age = _hit_age_stats(hits, cutoff)
    if age and age.get("n_with_date"):
        parts.append(
            f"hit-age stats (days from settle): "
            f"freshest={age['freshest_hit_age_days']} "
            f"median={age['median_hit_age_days']} "
            f"within_7d={age['hits_within_7d']} "
            f"within_30d={age['hits_within_30d']} "
            f"(of {age['n_with_date']}/{len(hits)} hits with parseable date)"
        )
    parts.append("")
    parts.append(f"# Retrieved articles (top-{len(hits)} by RRF)")
    parts.append("")
    if not hits:
        parts.append("(no articles were retrieved in this window)")
    else:
        for i, h in enumerate(hits, start=1):
            parts.append(_format_hit_block(h, i))
            parts.append("")

    return "\n".join(parts), len(hits)


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Parse a raw, fenced, or brace-delimited JSON response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate(parsed: dict[str, Any], retrieved: dict) -> dict[str, Any]:
    """Validate and normalize one judge response."""
    out: dict[str, Any] = {}

    v = parsed.get("verdict")
    if not isinstance(v, str) or v.upper() not in VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {v!r}")
    out["verdict"] = v.upper()

    inf = parsed.get("informativeness")
    if isinstance(inf, bool) or not isinstance(inf, int):
        raise ValueError(f"informativeness must be an integer: {inf!r}")
    if inf not in VALID_INFORMATIVENESS:
        raise ValueError(f"informativeness must be 0..3: {inf}")
    out["informativeness"] = inf

    # Record score/verdict disagreement without rejecting the response.
    expected = {0: "INSUFFICIENT", 1: "PARTIAL", 2: "SUFFICIENT", 3: "SUFFICIENT"}
    if expected.get(inf) != out["verdict"]:
        logger.debug(
            "verdict<->informativeness mismatch: %s vs %d (event=%s)",
            out["verdict"], inf, retrieved.get("event_id"),
        )

    lk = parsed.get("leakage_detected")
    if not isinstance(lk, bool):
        raise ValueError(f"leakage_detected must be bool: {lk!r}")
    out["leakage_detected"] = lk
    out["leakage_reason"] = parsed.get("leakage_reason") or None

    lean = parsed.get("confident_lean")
    if lean is None or (isinstance(lean, str) and lean.strip()):
        out["confident_lean"] = lean.strip() if isinstance(lean, str) else None
    else:
        raise ValueError(f"confident_lean must be null or non-empty string: {lean!r}")

    kh = parsed.get("key_hits") or []
    if not isinstance(kh, list):
        raise ValueError(f"key_hits must be a list: {kh!r}")
    cleaned: list[str] = []
    for x in kh:
        if isinstance(x, str) and x.strip():
            cleaned.append(x.strip())
        if len(cleaned) >= MAX_KEY_HITS:
            break
    out["key_hits"] = cleaned

    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("reasoning must be a non-empty string")
    out["reasoning"] = reasoning.strip()

    return out


def judge_question(
    *,
    q: dict,
    retrieved: dict,
    client: AnthropicLLMClient,
    system_prompt: str | None = None,
    max_hits: int = DEFAULT_MAX_HITS_IN_PROMPT,
) -> JudgeResult:
    """Judge one (question, retrieved) pair. Errors are recorded, not raised."""
    system_prompt = system_prompt or _load_system_prompt()
    user_prompt, n_hits = _format_user_prompt(q, retrieved, max_hits=max_hits)

    result = JudgeResult(
        event_id=str(q.get("event_id") or retrieved.get("event_id") or ""),
        slug=q.get("slug") or retrieved.get("slug") or "",
        title=q.get("title") or retrieved.get("title") or "",
        model_id=client.model_id,
        n_hits_seen=n_hits,
    )

    # Empty evidence sets need no model call.
    if n_hits == 0:
        result.verdict = "INSUFFICIENT"
        result.informativeness = 0
        result.leakage_detected = False
        result.leakage_reason = None
        result.confident_lean = None
        result.key_hits = []
        result.reasoning = (
            "No articles were retrieved in this question's pre-resolution "
            "window; the corpus carries no signal for this question."
        )
        return result

    try:
        text = client.chat(prompt=user_prompt, system_prompt=system_prompt)
    except Exception as e:
        result.error = f"chat_error: {type(e).__name__}: {e}"
        return result

    parsed = _extract_json(text)
    if parsed is None:
        result.error = f"unparseable_json (response head: {text[:200]!r})"
        return result

    try:
        validated = _validate(parsed, retrieved)
    except ValueError as e:
        result.error = f"schema_error: {e}"
        return result

    result.verdict = validated["verdict"]
    result.informativeness = validated["informativeness"]
    result.leakage_detected = validated["leakage_detected"]
    result.leakage_reason = validated["leakage_reason"]
    result.confident_lean = validated["confident_lean"]
    result.key_hits = validated["key_hits"]
    result.reasoning = validated["reasoning"]
    return result


def _load_completed_event_ids(output_path: Path) -> set[str]:
    """Load event IDs with error-free judge results."""
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            eid = rec.get("event_id")
            if eid:
                done.add(str(eid))
    return done


def _load_questions(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            eid = str(r.get("event_id") or "")
            if eid:
                out[eid] = r
    return out


def _iter_retrieved(path: Path):
    """Read all retrieval rows, including failures handled as zero-hit inputs."""
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--questions", type=Path,
        default=Path("questions/polymarket/filtered_signal.jsonl"),
        help="Source questions JSONL (for body / resolution criteria).",
    )
    ap.add_argument(
        "--retrieved", type=Path,
        default=Path("results/retrieved.jsonl"),
        help="Evidence JSONL from data.evidence_check.retrieve.",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("results/judged.jsonl"),
        help="Output JSONL; completed event IDs are skipped on resume.",
    )
    ap.add_argument(
        "--model-id", default=DEFAULT_JUDGE_MODEL,
        help=f"Claude model ID (default: {DEFAULT_JUDGE_MODEL}).",
    )

    ap.add_argument(
        "--max-tokens", type=int, default=8000,
        help="Maximum response tokens; must exceed --thinking-budget in legacy mode.",
    )
    ap.add_argument(
        "--thinking-schema", default="legacy",
        choices=["adaptive", "legacy"],
        help="adaptive uses --thinking-effort; legacy uses --thinking-budget.",
    )
    ap.add_argument(
        "--thinking-effort", default="low",
        choices=["low", "medium", "high", "max"],
        help="Adaptive-thinking effort (only used with --thinking-schema=adaptive).",
    )
    ap.add_argument(
        "--thinking-budget", type=int, default=6000,
        help="Legacy thinking tokens; must be below --max-tokens.",
    )
    ap.add_argument(
        "--no-thinking", action="store_true",
        help="Disable extended thinking.",
    )
    ap.add_argument(
        "--max-hits-in-prompt", type=int, default=DEFAULT_MAX_HITS_IN_PROMPT,
        help="Maximum retrieved articles included in the judge prompt.",
    )
    ap.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent judge calls.",
    )
    ap.add_argument(
        "--question-start", type=int, default=0,
        help="0-indexed first retrieval row to judge.",
    )
    ap.add_argument(
        "--question-end", type=int, default=None,
        help="0-indexed last retrieval row, exclusive.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Hard cap on total questions this run (post-resume-skip).",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    questions = _load_questions(args.questions)
    logger.info("loaded %d questions from %s", len(questions), args.questions)

    retrieved = list(_iter_retrieved(args.retrieved))
    logger.info("loaded %d retrieval rows from %s", len(retrieved), args.retrieved)

    end = args.question_end if args.question_end is not None else len(retrieved)
    rs = retrieved[args.question_start:end]
    logger.info("slice [%d, %d) = %d rows", args.question_start, end, len(rs))

    completed = _load_completed_event_ids(args.output)
    pending = [
        r for r in rs
        if str(r.get("event_id") or "") not in completed
        and str(r.get("event_id") or "") in questions
    ]
    logger.info("resume: %d already done; %d pending", len(completed), len(pending))
    if args.limit is not None:
        pending = pending[: args.limit]
        logger.info("--limit applied: capped pending to %d", len(pending))
    if not pending:
        logger.info("nothing to do.")
        return

    client = AnthropicLLMClient(
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        enable_thinking=not args.no_thinking,
        thinking_schema=args.thinking_schema,
        thinking_effort=args.thinking_effort,
        thinking_budget=args.thinking_budget,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_lock = threading.Lock()
    n_ok = 0
    n_err = 0
    n_suff = 0
    n_part = 0
    n_insuf = 0
    n_leakage = 0
    system_prompt = _load_system_prompt()

    def _one(retrieved_row: dict) -> JudgeResult:
        eid = str(retrieved_row.get("event_id") or "")
        q = questions.get(eid)
        if q is None:
            res = JudgeResult(
                event_id=eid,
                slug=retrieved_row.get("slug") or "",
                title=retrieved_row.get("title") or "",
                model_id=client.model_id,
                error="question_not_found",
            )
            return res
        return judge_question(
            q=q,
            retrieved=retrieved_row,
            client=client,
            system_prompt=system_prompt,
            max_hits=args.max_hits_in_prompt,
        )

    total = len(pending)
    initial = len(completed) if args.limit is None else 0
    bar_total = total + initial

    with client, args.output.open("a") as out_f, \
         ThreadPoolExecutor(max_workers=args.workers) as pool, \
         logging_redirect_tqdm(), \
         tqdm(
             total=bar_total,
             initial=initial,
             unit="q",
             desc=f"judge (workers={args.workers})",
             dynamic_ncols=True,
             smoothing=0.05,
         ) as bar:
        futs = {pool.submit(_one, r): r for r in pending}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                logger.exception("worker raised: %s", e)
                bar.update(1)
                continue
            with out_lock:
                out_f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
                out_f.flush()
                if res.error:
                    n_err += 1
                else:
                    n_ok += 1
                    if res.verdict == "SUFFICIENT":
                        n_suff += 1
                    elif res.verdict == "PARTIAL":
                        n_part += 1
                    elif res.verdict == "INSUFFICIENT":
                        n_insuf += 1
                    if res.leakage_detected:
                        n_leakage += 1
            bar.set_postfix(
                ok=n_ok, err=n_err,
                suf=n_suff, par=n_part, ins=n_insuf,
                leak=n_leakage,
                refresh=False,
            )
            bar.update(1)

    logger.info(
        "judge complete: ok=%d  err=%d  | SUFFICIENT=%d PARTIAL=%d INSUFFICIENT=%d  "
        "leakage=%d  out=%s",
        n_ok, n_err, n_suff, n_part, n_insuf, n_leakage, args.output,
    )


if __name__ == "__main__":
    main()
