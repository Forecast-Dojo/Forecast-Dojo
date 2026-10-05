"""Select questions with sufficient evidence and no flagged outcome leakage."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", required=True, help="filtered_signal.jsonl (full question records)")
    ap.add_argument("--judged", required=True, help="judged.jsonl (informativeness verdicts)")
    ap.add_argument("--output", required=True, help="final filtered JSONL to write")
    ap.add_argument("--keep-verdicts", default="SUFFICIENT",
                    help="comma-separated verdicts to keep (default: SUFFICIENT)")
    args = ap.parse_args(argv)

    keep_verdicts = {v.strip().upper() for v in args.keep_verdicts.split(",") if v.strip()}

    judged: Dict[str, Dict[str, Any]] = {}
    for j in _read_jsonl(args.judged):
        judged[str(j.get("event_id"))] = j

    n_in = 0
    n_kept = 0
    n_no_verdict = 0
    drop_verdict = 0
    drop_leakage = 0
    drop_error = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for rec in _read_jsonl(args.signal):
            n_in += 1
            eid = str(rec.get("event_id"))
            j = judged.get(eid)
            if j is None:
                n_no_verdict += 1
                continue
            if j.get("error"):
                drop_error += 1
                continue
            if j.get("leakage_detected") is True:
                drop_leakage += 1
                continue
            if str(j.get("verdict", "")).upper() not in keep_verdicts:
                drop_verdict += 1
                continue

            rec_out = dict(rec)
            rec_out["informativeness"] = {
                "verdict": j.get("verdict"),
                "score": j.get("informativeness"),
                "confident_lean": j.get("confident_lean"),
                "leakage_detected": j.get("leakage_detected"),
                "reasoning": j.get("reasoning"),
                "judge_model": j.get("model_id"),
            }
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n_kept += 1

    print(f"[select_informative] {args.signal}")
    print(f"  read              : {n_in}")
    print(f"  kept (verdict in {sorted(keep_verdicts)}, no leakage): {n_kept}")
    print(f"  dropped verdict   : {drop_verdict}")
    print(f"  dropped leakage   : {drop_leakage}")
    print(f"  dropped judge err : {drop_error}")
    if n_no_verdict:
        print(f"  WARNING no verdict: {n_no_verdict} (signal records with no judged row)")
    print(f"  -> {args.output} ({n_kept} questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
