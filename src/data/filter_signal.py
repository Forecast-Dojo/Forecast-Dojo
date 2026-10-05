"""Remove nearly certain or directionless market histories."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional


EPS = 1e-6
THRESHOLDS: Dict[str, float] = {
    "alpha_lo":     0.05,
    "coh_lo":       0.20,
    "abs_move_lo":  0.50,
}


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def _winner_prob_series(rec: Dict[str, Any]) -> List[Optional[float]]:
    """Per-day P(eventual winner). None on gap days."""
    di = rec.get("daily_index") or []
    n = len(di)
    kind = rec.get("belief_kind")

    if kind == "binary":
        yes_hist = rec.get("raw_yes_history") or {}
        if not yes_hist:
            return [None] * n
        label = next(iter(yes_hist))
        yes_vals = yes_hist[label]
        resolved = (rec.get("resolved_label") or "").lower()
        if resolved == "yes":
            return [None if v is None else float(v) for v in yes_vals]
        if resolved == "no":
            return [None if v is None else 1.0 - float(v) for v in yes_vals]
        return [None] * n

    if kind == "multi_neg_risk":
        norm = rec.get("normalized_history") or {}
        labels = list(norm.keys())
        win_idx = rec.get("winner_market_index")
        if win_idx is None or not (0 <= win_idx < len(labels)):
            return [None] * n
        return [None if v is None else float(v) for v in norm[labels[win_idx]]]

    return [None] * n


def _faithful_mask(rec: Dict[str, Any]) -> List[bool]:
    f = rec.get("faithfulness") or {}
    mask = f.get("per_day_faithful") or []
    di = rec.get("daily_index") or []
    if len(mask) != len(di):
        return [True] * len(di)
    return [bool(b) for b in mask]


def _alpha_oracle(probs: List[Optional[float]], mask: List[bool]) -> Optional[float]:
    vals: List[float] = []
    for p, ok in zip(probs, mask):
        if p is None or not ok:
            continue
        pp = max(EPS, min(1.0 - EPS, float(p)))
        vals.append(-math.log(pp))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _coherence_and_abs(
    probs: List[Optional[float]], mask: List[bool]
) -> tuple:
    """Return (signed coherence, total movement) over consecutive active days."""
    num = 0.0
    den = 0.0
    for i in range(1, len(probs)):
        a, b = probs[i - 1], probs[i]
        if a is None or b is None:
            continue
        if not (mask[i - 1] and mask[i]):
            continue
        d = float(b) - float(a)
        num += d
        den += abs(d)
    if den < EPS:
        return None, 0.0
    return num / den, den


def _evaluate(rec: Dict[str, Any]) -> Dict[str, Any]:
    probs = _winner_prob_series(rec)
    mask = _faithful_mask(rec)
    a = _alpha_oracle(probs, mask)
    c, abs_move = _coherence_and_abs(probs, mask)

    if a is None:
        return {"keep": False, "reason": "alpha_undefined",
                "alpha_oracle": None, "coh": c, "abs_move": abs_move}
    if c is None:
        return {"keep": False, "reason": "no_movement",
                "alpha_oracle": a, "coh": None, "abs_move": abs_move}

    fail_alpha_lo = a < THRESHOLDS["alpha_lo"]
    # Reject small movement only when it is also directionless.
    fail_structureless = (c < THRESHOLDS["coh_lo"]
                          and abs_move < THRESHOLDS["abs_move_lo"])
    keep = not (fail_alpha_lo or fail_structureless)

    reason = None
    if not keep:
        bits = []
        if fail_alpha_lo:      bits.append("alpha_below_lo")
        if fail_structureless: bits.append("structureless")
        reason = "+".join(bits)

    return {
        "keep": keep,
        "reason": reason,
        "alpha_oracle": a,
        "coh": c,
        "abs_move": abs_move,
        "thresholds": dict(THRESHOLDS),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True,
                    help="Faithfulness-filtered JSONL.")
    ap.add_argument("--output", required=True,
                    help="Output JSONL of events that pass the signal filter.")
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
    dropped_by_kind_reason: Counter = Counter()

    rej_f = open(args.rejected_output, "w", encoding="utf-8") if args.rejected_output else None
    with open(args.output, "w", encoding="utf-8") as out_f:
        for rec in _read_jsonl(args.input):
            n_in += 1
            kind = rec.get("belief_kind", "unknown")
            ev = _evaluate(rec)

            if ev["keep"]:
                rec_out = dict(rec)
                rec_out["learning_signal"] = {
                    "thresholds": ev["thresholds"],
                    "alpha_oracle": round(ev["alpha_oracle"], 5),
                    "coh":          round(ev["coh"], 5),
                    "abs_move":     round(ev["abs_move"], 5),
                }
                out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                n_kept += 1
                kept_by_kind[kind] += 1
            else:
                dropped_by_kind[kind] += 1
                dropped_by_reason[ev["reason"]] += 1
                dropped_by_kind_reason[(kind, ev["reason"])] += 1
                if rej_f is not None:
                    rec_out = dict(rec)
                    rec_out["learning_signal_reject"] = ev
                    rej_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    if rej_f is not None:
        rej_f.close()

    print(f"[filter_by_signal] read   : {n_in:>6}")
    print(f"[filter_by_signal] kept   : {n_kept:>6}")
    print(f"[filter_by_signal] dropped: {n_in - n_kept:>6}")
    print(f"[filter_by_signal] -> {args.output}")
    if args.rejected_output:
        print(f"[filter_by_signal] rejected -> {args.rejected_output}")

    print(f"\n[thresholds]")
    print(f"  alpha_oracle >= {THRESHOLDS['alpha_lo']}")
    print(f"  drop iff coh < {THRESHOLDS['coh_lo']} AND abs_move < {THRESHOLDS['abs_move_lo']}")

    print(f"\n[by belief_kind]")
    print(f"  {'kind':<18} {'kept':>8} {'dropped':>8} {'kept%':>8}")
    for k in sorted(set(kept_by_kind) | set(dropped_by_kind)):
        kept = kept_by_kind.get(k, 0)
        drop = dropped_by_kind.get(k, 0)
        tot = kept + drop
        pct = (kept / tot) if tot else 0.0
        print(f"  {k:<18} {kept:>8} {drop:>8} {pct:>8.1%}")

    if dropped_by_reason:
        print(f"\n[drop reasons (overall)]")
        for r, c in dropped_by_reason.most_common():
            print(f"  {r:<28} {c:>8}")

        print(f"\n[drop reasons by kind]")
        kinds_order = sorted({k for (k, _) in dropped_by_kind_reason})
        reasons_order = [r for r, _ in dropped_by_reason.most_common()]
        head = f"  {'reason':<28}  " + "  ".join(f"{k:>16}" for k in kinds_order)
        print(head)
        for r in reasons_order:
            bits = []
            for k in kinds_order:
                bits.append(f"{dropped_by_kind_reason.get((k, r), 0):>16}")
            print(f"  {r:<28}  " + "  ".join(bits))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
