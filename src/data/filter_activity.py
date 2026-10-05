"""Filter events using daily market activity and price coverage."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple


THRESHOLDS: Dict[str, Dict[str, float]] = {
    "binary":         {"tau_t": 6, "tau_v": 200.0, "s_max": 3, "rho_min": 0.80},
    "multi_neg_risk": {"tau_t": 5, "tau_v":  50.0, "s_max": 5, "rho_min": 0.70},
}
VOLUME_SOURCE = "winner_daily_volume"


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def _per_day_series(rec: Dict[str, Any]) -> Tuple[List[int], List[float]]:
    """Aligned (trades, share_volume) over ``daily_index`` from ``volume_source``."""
    di = rec.get("daily_index") or []
    series = rec.get(VOLUME_SOURCE) or rec.get("daily_volume") or []
    by_date: Dict[str, Dict[str, Any]] = {
        e["date"]: e for e in series if "date" in e
    }
    trades: List[int] = []
    shares: List[float] = []
    for d in di:
        e = by_date.get(d[:10]) or by_date.get(d) or {}
        trades.append(int(e.get("trades", 0) or 0))
        shares.append(float(e.get("share_volume", 0.0) or 0.0))
    return trades, shares


def _max_consecutive_false_run(mask: List[bool]) -> int:
    max_run = 0
    run = 0
    for v in mask:
        if v:
            run = 0
        else:
            run += 1
            if run > max_run:
                max_run = run
    return max_run


def _evaluate(rec: Dict[str, Any]) -> Dict[str, Any]:
    kind = rec.get("belief_kind", "")
    th = THRESHOLDS.get(kind)
    if th is None:
        return {"keep": False, "reason": f"unknown_kind:{kind}"}

    di = rec.get("daily_index") or []
    n = len(di)
    if n == 0:
        return {"keep": False, "reason": "no_daily_index"}

    trades, shares = _per_day_series(rec)
    faithful = [
        t >= th["tau_t"] and v >= th["tau_v"]
        for t, v in zip(trades, shares)
    ]
    n_faithful = sum(1 for f in faithful if f)
    frac = n_faithful / n
    max_run = _max_consecutive_false_run(faithful)

    keep = (max_run <= th["s_max"]) and (frac >= th["rho_min"])

    return {
        "keep": keep,
        "n_days": n,
        "n_faithful": n_faithful,
        "frac_faithful": round(frac, 4),
        "max_unfaithful_run": max_run,
        "per_day_faithful": faithful,
        "thresholds": {
            "tau_t": th["tau_t"],
            "tau_v": th["tau_v"],
            "s_max": th["s_max"],
            "rho_min": th["rho_min"],
        },
        "volume_source": VOLUME_SOURCE,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Lifetime-filtered JSONL.")
    ap.add_argument("--output", required=True,
                    help="Output JSONL of events that pass faithfulness.")
    ap.add_argument("--rejected-output", default=None,
                    help="Optional path to write rejected records (with reason).")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)

    n_in = 0
    n_kept = 0
    kept_by_kind: Counter = Counter()
    dropped_by_kind: Counter = Counter()
    dropped_by_reason: Counter = Counter()

    rej_f = open(args.rejected_output, "w", encoding="utf-8") if args.rejected_output else None
    with open(args.output, "w", encoding="utf-8") as out_f:
        for rec in _read_jsonl(args.input):
            n_in += 1
            kind = rec.get("belief_kind", "unknown")
            ev = _evaluate(rec)

            if ev["keep"]:
                rec_out = dict(rec)
                rec_out["faithfulness"] = {
                    k: ev[k] for k in (
                        "thresholds", "volume_source", "n_days",
                        "n_faithful", "frac_faithful", "max_unfaithful_run",
                        "per_day_faithful",
                    )
                }
                out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                n_kept += 1
                kept_by_kind[kind] += 1
            else:
                dropped_by_kind[kind] += 1
                if "reason" in ev:
                    dropped_by_reason[ev["reason"]] += 1
                else:
                    th = ev["thresholds"]
                    fail_run = ev["max_unfaithful_run"] > th["s_max"]
                    fail_rho = ev["frac_faithful"] < th["rho_min"]
                    if fail_run and fail_rho:
                        dropped_by_reason["both"] += 1
                    elif fail_run:
                        dropped_by_reason["s_max_exceeded"] += 1
                    else:
                        dropped_by_reason["rho_min_below"] += 1
                if rej_f is not None:
                    rec_out = dict(rec)
                    rec_out["faithfulness_reject"] = ev
                    rej_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    if rej_f is not None:
        rej_f.close()

    print(f"[filter_by_faithfulness] read   : {n_in:>6}")
    print(f"[filter_by_faithfulness] kept   : {n_kept:>6}")
    print(f"[filter_by_faithfulness] dropped: {n_in - n_kept:>6}")
    print(f"[filter_by_faithfulness] -> {args.output}")
    if args.rejected_output:
        print(f"[filter_by_faithfulness] rejected -> {args.rejected_output}")

    print("\n[by belief_kind]")
    print(f"  {'kind':<18} {'kept':>8} {'dropped':>8} {'kept%':>8}")
    for k in sorted(set(kept_by_kind) | set(dropped_by_kind)):
        kept = kept_by_kind.get(k, 0)
        drop = dropped_by_kind.get(k, 0)
        tot = kept + drop
        pct = (kept / tot) if tot else 0.0
        print(f"  {k:<18} {kept:>8} {drop:>8} {pct:>8.1%}")

    if dropped_by_reason:
        print("\n[drop reasons]")
        for r, c in dropped_by_reason.most_common():
            print(f"  {r:<22} {c:>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
