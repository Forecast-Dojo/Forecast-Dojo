"""Select events with sufficiently long trading histories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def _lifetime_days(record: Dict[str, Any]) -> int:
    di = record.get("daily_index") or []
    return len(di)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Filter Polymarket events by tradeable-lifetime length."
    )
    ap.add_argument(
        "--input", required=True,
        help="Trading-activity JSONL from data.collect_volume.",
    )
    ap.add_argument(
        "--output", required=True,
        help="Path to write the filtered JSONL.",
    )
    ap.add_argument(
        "--min-days", type=int, default=8,
        help=(
            "Minimum UTC calendar days in daily_index (default: 8)."
        ),
    )
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    n_in = 0
    n_kept = 0
    days_hist: Counter = Counter()
    kind_kept: Counter = Counter()
    kind_dropped: Counter = Counter()

    with open(args.output, "w", encoding="utf-8") as out_f:
        for rec in _read_jsonl(args.input):
            n_in += 1
            d = _lifetime_days(rec)
            days_hist[d] += 1
            kind = rec.get("belief_kind", "unknown")
            if d >= args.min_days:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
                kind_kept[kind] += 1
            else:
                kind_dropped[kind] += 1

    print(f"[filter_by_lifetime] read   : {n_in:>6} events from {args.input}")
    print(f"[filter_by_lifetime] kept   : {n_kept:>6} (lifetime >= {args.min_days} days)")
    print(f"[filter_by_lifetime] dropped: {n_in - n_kept:>6}")
    print(f"[filter_by_lifetime] -> {args.output}")

    if kind_kept or kind_dropped:
        print("\n[by belief_kind]")
        all_kinds = sorted(set(kind_kept) | set(kind_dropped))
        print(f"  {'kind':<18} {'kept':>8} {'dropped':>8}")
        for k in all_kinds:
            print(f"  {k:<18} {kind_kept.get(k,0):>8} {kind_dropped.get(k,0):>8}")

    if days_hist:
        print("\n[lifetime distribution buckets]")
        buckets = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30),
                   (31, 60), (61, 120), (121, 10**9)]
        labels = ["1d", "2-3d", "4-7d", "8-14d", "15-30d",
                  "31-60d", "61-120d", "121d+"]
        for (lo, hi), label in zip(buckets, labels):
            n = sum(c for d, c in days_hist.items() if lo <= d <= hi)
            print(f"  {label:>8} : {n:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
