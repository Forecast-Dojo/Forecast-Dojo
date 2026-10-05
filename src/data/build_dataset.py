# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build dated forecasting tasks from curated event records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .forecast_dates import crowd_probs_for_dates, pick_forecast_dates
from .placeholder_filter import filter_question

# Shards are named YYYY-MM or YYYY-MM-DD.
_SHARD_NAME_RE = re.compile(r"^\d{4}-\d{2}(?:-\d{2})?$")


def earliest_indexed_date(index_root: str | Path) -> str | None:
    """Earliest shard date under `index_root` as `YYYY-MM-DD` (month shards -> day 01)."""
    root = Path(index_root)
    if not root.is_dir():
        return None
    candidates = sorted(d.name for d in root.iterdir() if d.is_dir() and _SHARD_NAME_RE.match(d.name))
    if not candidates:
        return None
    earliest = candidates[0]
    return earliest if len(earliest) == 10 else f"{earliest}-01"


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def compute_global_daily_volume(index_root: str | Path, *, parquet_name: str = "texts.parquet") -> dict[str, int]:
    """Count indexed articles by publication date across all corpus shards."""
    import pyarrow.parquet as pq

    root = Path(index_root)
    shards = sorted(d for d in root.iterdir() if d.is_dir() and _SHARD_NAME_RE.match(d.name))
    if not shards:
        raise SystemExit(f"--news-volume: no `YYYY-MM[-DD]` shard dirs under {root}")

    counts: dict[str, int] = {}
    n_rows = n_bad = 0
    for shard in shards:
        parquet = shard / parquet_name
        if not parquet.exists() or "published_date" not in set(pq.read_schema(parquet).names):
            print(f"# news-volume: skip {shard.name} (no {parquet_name}/published_date)")
            continue
        for batch in pq.ParquetFile(parquet).iter_batches(columns=["published_date"]):
            for v in batch.column(0).to_pylist():
                n_rows += 1
                d = str(v)[:10] if v else ""
                if _DATE_RE.match(d):
                    counts[d] = counts.get(d, 0) + 1
                else:
                    n_bad += 1
    print(f"# news-volume: {len(counts)} dated days from {len(shards)} shard(s) "
          f"({n_rows} rows, {n_bad} undated)")
    return counts


def load_global_daily_volume(path: Path) -> dict[str, int]:
    """Load a precomputed {YYYY-MM-DD: article_count} histogram."""
    raw = json.loads(path.read_text())
    return {str(k)[:10]: int(v) for k, v in raw.items()}


def attach_news_daily(q: dict, global_volume: dict[str, int]) -> None:
    """Set `q["news_daily"]`: global per-day article count sliced to `daily_index`."""
    q["news_daily"] = [global_volume.get(d[:10], 0) for d in (q.get("daily_index") or [])]


def _question_payload(q: dict) -> dict[str, Any]:
    """Extract question fields for rendering forecast prompts."""
    # Use recorded close time rather than the creator-set deadline.
    payload: dict[str, Any] = {
        "title": q["title"],
        "body": (q.get("body") or "").strip(),
        "close_date": (q.get("close_date") or "")[:10],
        "belief_kind": q["belief_kind"],
    }
    if q["belief_kind"] != "binary":
        payload["markets"] = [
            {"label": m["label"], "question": m.get("question") or ""}
            for m in q.get("markets") or []
        ]
    return payload


def _resolved_truth(q: dict) -> str:
    """Episode ground-truth label, upper-cased to YES/NO for binary questions."""
    truth = (q.get("resolved_label") or "").strip()
    if q["belief_kind"] == "binary" and truth.lower() in ("yes", "no"):
        return truth.upper()
    return truth


def build_rows(
    questions: list[dict],
    *,
    global_start: str | None,
    auto_k: bool,
    fixed_k: int | None,
    alpha: float,
    k_min: int,
    k_max: int,
    c: float,
    min_days_before_resolution: int,
    mode: str = "memory-on",
) -> list[dict]:
    """Build event-level memory-on or date-level memory-free tasks."""
    rows: list[dict] = []
    for q in questions:
        if fixed_k is not None:
            dates = pick_forecast_dates(
                q, k=fixed_k, alpha=alpha,
                min_days_before_resolution=min_days_before_resolution,
            )
        else:
            dates = pick_forecast_dates(
                q, alpha=alpha, auto_k=True, k_min=k_min, k_max=k_max, c=c,
                min_days_before_resolution=min_days_before_resolution,
            )
        if not dates:
            continue

        crowd_probs = crowd_probs_for_dates(q, dates)

        start_date = global_start or q["start_date"][:10]
        # Rollout fields belong in metadata; prompt is used for length accounting.
        base = {
            "event_id": q.get("event_id"),
            "slug": q.get("slug"),
            "belief_kind": q["belief_kind"],
            "label": _resolved_truth(q),
            "prompt": f"Forecast: {q['title']}",
        }
        question_payload = _question_payload(q)
        if mode == "memory-free":
            for d, cp in zip(dates, crowd_probs):
                rows.append({
                    **base,
                    "metadata": {
                        "start_date": start_date,
                        "forecast_date": d,
                        "crowd_prob": cp,
                        "question": question_payload,
                    },
                })
        else:
            rows.append({
                **base,
                "metadata": {
                    "start_date": start_date,
                    "forecast_dates": dates,
                    "crowd_probs": crowd_probs,
                    "question": question_payload,
                },
            })
    return rows


def sort_rows(rows: list[dict], *, by: str) -> list[dict]:
    """Sort tasks by entry, resolution, or forecast date; "none" preserves order."""
    if by == "none":
        return rows

    def key(row: dict) -> tuple[str, str, str]:
        meta = row["metadata"]
        eid = str(row.get("event_id"))
        if by == "forecast_date":
            return (meta["forecast_date"], eid, "")
        fds = meta["forecast_dates"]
        if by == "entry":
            return (fds[0], fds[-1], eid)
        # Missing close dates fall back to the last forecast date.
        return (meta["question"].get("close_date") or fds[-1], fds[-1], eid)

    return sorted(rows, key=key)


def load_questions(path: Path) -> list[dict]:
    """Read the questions JSONL, applying the placeholder-leg filter."""
    questions: list[dict] = []
    n_dropped_legs = 0
    n_skipped = 0
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            raw = json.loads(ln)
            cleaned = filter_question(raw)
            if cleaned is None:
                n_skipped += 1
                continue
            if cleaned is not raw:
                n_dropped_legs += len(raw.get("markets") or []) - len(cleaned.get("markets") or [])
            questions.append(cleaned)
    if n_dropped_legs or n_skipped:
        print(f"# placeholder filter: dropped {n_dropped_legs} legs; skipped {n_skipped} broken question(s)")
    return questions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Curated question JSONL.")
    ap.add_argument("--output", type=Path, required=True, help="Output task JSONL.")
    ap.add_argument("--news-volume", action="store_true",
                    help="Count corpus articles per day using --index-root; otherwise use belief movement only.")
    ap.add_argument("--news-volume-json", type=Path, default=None,
                    help="Load precomputed {YYYY-MM-DD: article_count} values instead of scanning the corpus.")
    ap.add_argument("--index-root", default=None,
                    help="FAISS index root; its earliest shard sets the default search-window start.")
    ap.add_argument("--start-date", default=None,
                    help="Override the search-window lower bound (ISO YYYY-MM-DD).")
    ap.add_argument("--auto-k", action="store_true", help="Sub-linear K(H) = clamp(round(c*sqrt(H)), k_min, k_max).")
    ap.add_argument("--fixed-k", type=int, default=None, help="Constant K per question (overrides --auto-k).")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--c", type=float, default=1.0, help="Slope on sqrt(H) for --auto-k.")
    ap.add_argument("--alpha", type=float, default=0.7, help="Picker weight on belief delta vs. volume.")
    ap.add_argument("--min-days-before-resolution", type=int, default=2,
                    help="Leakage buffer: exclude forecast dates within this many days of resolution.")
    ap.add_argument("--mode", choices=("memory-on", "memory-free"), default="memory-on",
                    help="memory-on: one event per row; memory-free: one event-date per row.")
    ap.add_argument("--sort-by", choices=("none", "entry", "resolution", "forecast_date"), default=None,
                    help="Row order: none by default for memory-on, forecast_date for memory-free. Entry/resolution require memory-on.")
    args = ap.parse_args()

    if args.fixed_k is None and not args.auto_k:
        ap.error("pass --auto-k or --fixed-k")
    if args.sort_by is None:
        args.sort_by = "forecast_date" if args.mode == "memory-free" else "none"
    if args.mode == "memory-free" and args.sort_by in ("entry", "resolution"):
        ap.error("--sort-by entry/resolution are memory-on-only; use forecast_date or none")
    if args.mode == "memory-on" and args.sort_by == "forecast_date":
        ap.error("--sort-by forecast_date is memory-free-only; use entry/resolution or none")

    if args.start_date:
        global_start = args.start_date
        print(f"# search-window start: {global_start} (CLI override)")
    elif args.index_root:
        global_start = earliest_indexed_date(args.index_root)
        print(f"# search-window start: {global_start} (earliest shard under {args.index_root})")
    else:
        global_start = None
        print("# search-window start: per-question start_date (no --index-root / --start-date given)")

    questions = load_questions(args.input)
    print(f"# loaded {len(questions)} questions from {args.input}")

    if args.news_volume_json:
        global_volume = load_global_daily_volume(args.news_volume_json)
        print(f"# news volume: loaded {len(global_volume)} dated days from {args.news_volume_json}")
    elif args.news_volume:
        if not args.index_root:
            ap.error("--news-volume needs --index-root (the corpus location)")
        global_volume = compute_global_daily_volume(args.index_root)
    else:
        global_volume = None
        print("# news volume: OFF -> picker uses belief movement only")

    if global_volume is not None:
        for q in questions:
            attach_news_daily(q, global_volume)
        n_with = sum(1 for q in questions if any(q.get("news_daily") or []))
        print(f"# news volume: attached ({n_with}/{len(questions)} questions overlap >=1 article-day)")

    rows = build_rows(
        questions,
        global_start=global_start,
        auto_k=args.auto_k,
        fixed_k=args.fixed_k,
        alpha=args.alpha,
        k_min=args.k_min,
        k_max=args.k_max,
        c=args.c,
        min_days_before_resolution=args.min_days_before_resolution,
        mode=args.mode,
    )

    rows = sort_rows(rows, by=args.sort_by)
    if args.sort_by != "none":
        print(f"# sorted {len(rows)} episodes by '{args.sort_by}' "
              f"(remember: needs --rollout-shuffle OFF in the launcher to take effect)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.mode == "memory-free":
        print(f"# wrote {len(rows)} memory-free (question, forecast_date) rows -> {args.output}")
    else:
        total_steps = sum(len(r["metadata"]["forecast_dates"]) for r in rows)
        print(f"# wrote {len(rows)} episodes ({total_steps} total steps, "
              f"avg {total_steps / max(len(rows), 1):.2f}/episode) -> {args.output}")


if __name__ == "__main__":
    main()
