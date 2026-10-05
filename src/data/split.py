"""Split complete event lifetimes into non-overlapping date windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


def _ts(x: Any) -> Optional[pd.Timestamp]:
    if not x:
        return None
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return None


def _event_span(rec: Dict[str, Any]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return the earliest market open and latest market close across supported date fields."""
    markets = rec.get("markets") or []
    starts, closes = [], []
    for m in markets:
        o = _ts(m.get("start_date")) or _ts(m.get("open_time"))
        c = _ts(m.get("closed_time")) or _ts(m.get("close_time"))
        if o is not None:
            starts.append(o)
        if c is not None:
            closes.append(c)
    if not starts or not closes:
        return None, None
    return min(starts), max(closes)


def _parse_window(spec: str) -> Tuple[str, pd.Timestamp, pd.Timestamp, str]:
    """Parse 'name:START:END:outpath' (dates YYYY-MM-DD; window is [START, END))."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"--window must be name:START:END:outpath, got {spec!r}"
        )
    name, start, end, out = parts
    return name, pd.Timestamp(start + "T00:00:00Z"), pd.Timestamp(end + "T00:00:00Z"), out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Split full.jsonl into strict time windows.")
    ap.add_argument("--input", required=True, help="Event JSONL with market opening and closing times.")
    ap.add_argument(
        "--window", action="append", required=True, dest="windows",
        help="name:START:END:outpath (YYYY-MM-DD; half-open [START,END)). Repeatable.",
    )
    args = ap.parse_args(argv)

    windows = [_parse_window(w) for w in args.windows]
    for _, _, _, out in windows:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    handles = {name: open(out, "w", encoding="utf-8") for name, _, _, out in windows}
    counts = {name: 0 for name, _, _, _ in windows}
    n_in = 0
    n_no_span = 0
    n_unassigned = 0  # Boundary-spanning events remain unassigned.

    total_bytes = os.path.getsize(args.input) if os.path.exists(args.input) else 0
    bar = tqdm(total=total_bytes, desc="Splitting", unit="B", unit_scale=True)
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for ln in f:
                bar.update(len(ln.encode("utf-8")))
                ln = ln.strip()
                if not ln:
                    continue
                n_in += 1
                rec = json.loads(ln)
                lo, hi = _event_span(rec)
                if lo is None or hi is None:
                    n_no_span += 1
                    continue
                placed = False
                for name, ws, we, _ in windows:
                    if lo >= ws and hi < we:
                        handles[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts[name] += 1
                        placed = True
                        break
                if not placed:
                    n_unassigned += 1
                bar.set_postfix(events=n_in, **counts)
    finally:
        bar.close()
        for h in handles.values():
            h.close()

    print(f"[split_by_window] read {n_in} events from {args.input}")
    for name, ws, we, out in windows:
        print(f"  {name:<8} [{ws.date()}, {we.date()}) -> {counts[name]:>6}  {out}")
    if n_no_span:
        print(f"  (dropped {n_no_span} with no usable start/close dates)")
    if n_unassigned:
        print(f"  (dropped {n_unassigned} boundary-spanning events — open in one "
              f"window, close in another; belong to neither strict window)")
    assigned = sum(counts.values())
    print(f"  total assigned {assigned} / {n_in}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
