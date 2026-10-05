"""Generate research queries for event evidence screening."""

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

DEFAULT_DECOMPOSE_MODEL = "claude-opus-4-8"

DECOMPOSE_PROMPT_PATH = PROMPTS_DIR / "evidence_check" / "decompose.md"

ANGLE_KEYS: tuple[str, ...] = ("evidence", "context")
MIN_QUERIES_PER_ANGLE = 4
MAX_QUERIES_PER_ANGLE = 7


@dataclass
class DecompositionResult:
    """Research queries and model metadata for one event."""

    event_id: str
    slug: str
    title: str
    belief_kind: str
    queries: dict[str, list[str]] = field(default_factory=dict)
    flat_queries: list[str] = field(default_factory=list)
    model_id: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "slug": self.slug,
            "title": self.title,
            "belief_kind": self.belief_kind,
            "queries": self.queries,
            "flat_queries": self.flat_queries,
            "model_id": self.model_id,
            "error": self.error,
        }


def _load_system_prompt() -> str:
    return DECOMPOSE_PROMPT_PATH.read_text(encoding="utf-8")


def _format_user_prompt(q: dict) -> str:
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


def _validate_and_normalize(parsed: dict[str, Any]) -> dict[str, list[str]]:
    """Validate and normalize the evidence and context query lists."""
    out: dict[str, list[str]] = {}
    for angle in ANGLE_KEYS:
        v = parsed.get(angle)
        if not isinstance(v, list):
            raise ValueError(f"missing or non-list angle: {angle!r}")
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if len(cleaned) < MIN_QUERIES_PER_ANGLE:
            raise ValueError(
                f"angle {angle!r} has {len(cleaned)} queries; "
                f"need >= {MIN_QUERIES_PER_ANGLE}"
            )
        out[angle] = cleaned[:MAX_QUERIES_PER_ANGLE]
    return out


def _flatten_queries(by_angle: dict[str, list[str]]) -> list[str]:
    """Deduplicate queries across angles, preserving order."""
    seen: set[str] = set()
    flat: list[str] = []
    for angle in ANGLE_KEYS:
        for q in by_angle.get(angle, []):
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            flat.append(q)
    return flat


def decompose_question(
    q: dict,
    client: AnthropicLLMClient,
    *,
    system_prompt: str | None = None,
) -> DecompositionResult:
    """Decompose one question. Returns a result either way (errors recorded)."""
    system_prompt = system_prompt or _load_system_prompt()
    user_prompt = _format_user_prompt(q)

    result = DecompositionResult(
        event_id=str(q.get("event_id") or ""),
        slug=q.get("slug") or "",
        title=q.get("title") or "",
        belief_kind=q.get("belief_kind") or "binary",
        model_id=client.model_id,
    )

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
        normalized = _validate_and_normalize(parsed)
    except ValueError as e:
        result.error = f"schema_error: {e}"
        return result

    result.queries = normalized
    result.flat_queries = _flatten_queries(normalized)
    return result


def decompose_questions(
    questions: list[dict],
    client: AnthropicLLMClient,
    *,
    workers: int = 8,
    progress_cb=None,
) -> list[DecompositionResult]:
    """Decompose in parallel, preserving input order and recording call errors."""
    system_prompt = _load_system_prompt()
    results: list[DecompositionResult | None] = [None] * len(questions)
    lock = threading.Lock()
    n_done = 0

    def _one(idx: int) -> tuple[int, DecompositionResult]:
        return idx, decompose_question(
            questions[idx], client, system_prompt=system_prompt,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, i) for i in range(len(questions))]
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res
            with lock:
                n_done += 1
                if progress_cb is not None:
                    progress_cb(n_done, len(questions), res)

    return [r for r in results if r is not None]


def _load_completed_ids(output_path: Path) -> set[str]:
    """Load event IDs with error-free decomposition results."""
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


def _iter_records(path: Path):
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input", type=Path,
        default=Path("questions/polymarket/filtered_signal.jsonl"),
        help="Questions JSONL (one record per line).",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("questions/polymarket/decomposed.jsonl"),
        help="Output JSONL; completed event IDs are skipped on resume.",
    )
    ap.add_argument(
        "--model-id", default=DEFAULT_DECOMPOSE_MODEL,
        help=f"Claude model ID (default: {DEFAULT_DECOMPOSE_MODEL}).",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Maximum response tokens, including thinking.",
    )
    ap.add_argument(
        "--thinking-schema", default="adaptive",
        choices=["adaptive", "legacy"],
        help="adaptive uses --thinking-effort; legacy uses --thinking-budget.",
    )
    ap.add_argument(
        "--thinking-effort", default="low",
        choices=["low", "medium", "high", "max"],
        help="Adaptive thinking effort (default: low).",
    )
    ap.add_argument(
        "--thinking-budget", type=int, default=2000,
        help="Legacy-thinking budget (tokens). Ignored for "
             "--thinking-schema=adaptive.",
    )
    ap.add_argument(
        "--no-thinking", action="store_true",
        help="Disable extended thinking.",
    )
    ap.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent decomposition calls.",
    )
    ap.add_argument(
        "--question-start", type=int, default=0,
        help="0-indexed first question to process (for sharded runs).",
    )
    ap.add_argument(
        "--question-end", type=int, default=None,
        help="0-indexed last question (exclusive). Default: end of file.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Maximum questions after resume filtering.",
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

    all_qs = list(_iter_records(args.input))
    end = args.question_end if args.question_end is not None else len(all_qs)
    qs_slice = all_qs[args.question_start:end]
    logger.info("loaded %d questions; processing slice [%d, %d) = %d",
                len(all_qs), args.question_start, end, len(qs_slice))

    completed = _load_completed_ids(args.output)
    pending = [q for q in qs_slice if str(q.get("event_id") or "") not in completed]
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

    # Persist each completed call so interrupted runs can resume.
    system_prompt = _load_system_prompt()

    def _one(q: dict) -> DecompositionResult:
        return decompose_question(q, client, system_prompt=system_prompt)

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
             desc=f"decompose (workers={args.workers})",
             dynamic_ncols=True,
             smoothing=0.05,
         ) as bar:
        futs = {pool.submit(_one, q): q for q in pending}
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
            bar.set_postfix(ok=n_ok, err=n_err, refresh=False)
            bar.update(1)

    logger.info("decomposition complete: ok=%d  err=%d  out=%s",
                n_ok, n_err, args.output)


if __name__ == "__main__":
    main()
