"""Forecast scores and event-bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

EPS = 1e-3
SCORES = ("brier", "accuracy", "log_loss", "info_alpha")


def outcome_labels(question: dict[str, Any]) -> list[str]:
    if question["belief_kind"] == "binary":
        return ["YES", "NO"]
    labels = list(dict.fromkeys(market["label"] for market in question.get("markets", [])))
    if not labels:
        raise ValueError("A question must have distinct offered outcome labels")
    return labels


def normalize_forecast(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or not value:
        return None
    if any(
        not isinstance(label, str) or not _finite(probability)
        for label, probability in value.items()
    ):
        return None
    clipped = {label: max(0.0, float(probability)) for label, probability in value.items()}
    scale = max(clipped.values())
    if scale <= 0:
        return None
    # Scale first to avoid overflow when summing finite values.
    total = math.fsum(probability / scale for probability in clipped.values())
    return {label: (probability / scale) / total for label, probability in clipped.items()}


def score_forecast(
    forecast: Any,
    truth: str,
    labels: list[str],
    *,
    parse_ok: bool = True,
    crowd_prob: float | None = None,
) -> dict[str, Any]:
    """Use uniform fallback for unusable reports; score unknown labels as error."""
    if not labels or truth not in labels:
        raise ValueError("The resolved label must be one of the offered outcomes")
    probabilities = normalize_forecast(forecast) if parse_ok else None
    usable = probabilities is not None
    if probabilities is None:
        probabilities = {label: 1.0 / len(labels) for label in labels}
    p_truth = probabilities.get(truth, 0.0)
    top = max(probabilities.values())
    tied = [label for label, probability in probabilities.items() if probability == top]
    accuracy = (1.0 / len(tied)) if truth in tied else 0.0
    brier = math.fsum(
        (probabilities.get(label, 0.0) - (label == truth)) ** 2
        for label in set(probabilities) | {truth}
    )
    log_loss = -math.log(max(p_truth, EPS)) if usable else math.log(len(labels))
    alpha = None
    if _finite(crowd_prob):
        alpha = math.log(max(p_truth, EPS)) - math.log(max(float(crowd_prob), EPS))
    return {
        "usable": usable, "brier": brier, "accuracy": accuracy, "log_loss": log_loss,
        "info_alpha": alpha, "p_truth": p_truth, "confidence": top,
        "probabilities": probabilities,
    }


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["event_id"]), row["forecast_date"], int(row.get("rollout_idx", 0))


def _tasks(dataset: str | Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    tasks = {}
    if dataset is None:
        return tasks
    for event in _read_rows(dataset):
        metadata = event["metadata"]
        if "forecast_dates" in metadata:
            dates = metadata["forecast_dates"]
            crowd = metadata.get("crowd_probs") or [None] * len(dates)
            vectors = metadata.get("crowd_distributions") or [None] * len(dates)
        else:
            dates = [metadata["forecast_date"]]
            crowd = [metadata.get("crowd_prob")]
            vectors = [metadata.get("crowd_distribution")]
        labels = outcome_labels(metadata["question"])
        if len(crowd) != len(dates) or len(vectors) != len(dates):
            raise ValueError("Market references must align with the forecast dates")
        for date, probability, vector in zip(dates, crowd, vectors):
            key = str(event["event_id"]), date
            if key in tasks:
                raise ValueError(f"Duplicate scheduled event-date: {key}")
            tasks[key] = {
                "labels": labels, "truth": event["label"], "crowd_prob": probability,
                "crowd_distribution": vector,
            }
    return tasks


def scored_records(
    path: str | Path, dataset: str | Path | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Score recorded forecasts, counting infrastructure errors as missing."""
    tasks = _tasks(dataset)
    records, errors, seen = [], 0, set()
    for row in _read_rows(path):
        if "error" in row:
            errors += 1
            continue
        key = _key(row)
        if key in seen:
            raise ValueError(f"Duplicate forecast record: {key}")
        seen.add(key)
        task = tasks.get(key[:2])
        if tasks and task is None:
            raise ValueError(f"Forecast is outside the supplied dataset: {key}")
        labels = task["labels"] if task else row.get("outcome_labels")
        if not labels:
            raise ValueError("Scoring requires offered outcome labels: pass --dataset for older records")
        truth = task["truth"] if task else row["resolved_label"]
        crowd = task["crowd_prob"] if task else row.get("crowd_prob")
        scores = score_forecast(
            row.get("agent_forecast"), truth, labels,
            parse_ok=row.get("parse_ok") is not False, crowd_prob=crowd,
        )
        records.append({**scores, "event_id": key[0], "forecast_date": key[1], "rollout_idx": key[2]})
    return records, errors


def calibration_error(records: list[dict[str, Any]], bins: int = 15) -> float | None:
    """Top-label ECE over usable reports, with fractional credit for tied predictions."""
    groups = defaultdict(list)
    for record in records:
        if record["usable"]:
            groups[min(int(record["confidence"] * bins), bins - 1)].append(record)
    count = sum(map(len, groups.values()))
    if not count:
        return None
    return sum(
        len(group) * abs(
            fmean(row["confidence"] for row in group) -
            fmean(row["accuracy"] for row in group)
        )
        for group in groups.values()
    ) / count


def _percentile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def event_bootstrap(
    records: list[dict[str, Any]],
    metric: str,
    *,
    samples: int = 4000,
    seed: int = 0,
) -> list[float] | None:
    """Resample events, preserving forecast-level weighting."""
    if samples <= 0:
        return None
    grouped = defaultdict(list)
    for record in records:
        if _finite(record.get(metric)):
            grouped[record["event_id"]].append(record[metric])
    totals = [(math.fsum(values), len(values)) for _, values in sorted(grouped.items())]
    if not totals:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = rng.choices(totals, k=len(totals))
        means.append(math.fsum(value for value, count in draw) / sum(count for value, count in draw))
    means.sort()
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def compare_runs(
    first: str | Path,
    second: str | Path,
    *,
    dataset: str | Path,
    pair_rollouts: bool = False,
    bootstrap: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return second-minus-first differences; pair rollout IDs for memory comparisons."""
    first_rows, _ = scored_records(first, dataset)
    second_rows, _ = scored_records(second, dataset)
    result = {}
    for metric in SCORES:
        def grouped(rows):
            groups = defaultdict(list)
            for row in rows:
                if _finite(row.get(metric)):
                    key = _key(row) if pair_rollouts else _key(row)[:2]
                    groups[key].append(row[metric])
            return {key: fmean(values) for key, values in groups.items()}

        left, right = grouped(first_rows), grouped(second_rows)
        differences = [
            {"event_id": key[0], metric: right[key] - left[key]}
            for key in sorted(left.keys() & right.keys())
        ]
        result[metric] = {
            "difference": _mean([row[metric] for row in differences]),
            "n_pairs": len(differences),
            "ci95": event_bootstrap(differences, metric, samples=bootstrap, seed=seed),
        }
    return result


def score_run(
    path: str | Path,
    *,
    dataset: str | Path | None = None,
    n_rollouts: int = 4,
    bootstrap: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    records, errors = scored_records(path, dataset)
    tasks = _tasks(dataset)
    expected = {(event, date, rollout) for event, date in tasks for rollout in range(n_rollouts)}
    observed = {_key(row) for row in records}
    if expected and observed - expected:
        raise ValueError("A record has a rollout index outside the scheduled range")
    missing = len(expected - observed) if expected else None
    unusable = sum(not row["usable"] for row in records)
    card = {
        "n_recorded": len(records), "n_usable": len(records) - unusable,
        "n_unusable": unusable, "n_errors": errors,
        "n_scheduled": len(expected) if tasks else None,
        "n_missing": missing,
        "failure_rate": (unusable + missing) / len(expected) if expected else None,
        "ece": calibration_error(records), "metrics": {},
    }
    for metric in SCORES:
        values = [row[metric] for row in records if _finite(row.get(metric))]
        rollouts = defaultdict(list)
        for row in records:
            if _finite(row.get(metric)):
                rollouts[row["rollout_idx"]].append(row[metric])
        rollout_means = [fmean(values) for _, values in sorted(rollouts.items())]
        card["metrics"][metric] = {
            "mean": _mean(values), "n": len(values),
            "rollout_std": pstdev(rollout_means) if rollout_means else None,
            "ci95": event_bootstrap(records, metric, samples=bootstrap, seed=seed),
        }
    if tasks:
        uniform, market = [], []
        for (event, date), task in tasks.items():
            uniform.append(score_forecast(None, task["truth"], task["labels"]))
            if normalize_forecast(task["crowd_distribution"]) is not None:
                market.append(score_forecast(task["crowd_distribution"], task["truth"], task["labels"]))
        card["references"] = {
            name: {
                "n_dates": len(rows),
                **{
                    metric: _mean([row[metric] for row in rows])
                    for metric in ("brier", "accuracy", "log_loss")
                },
            }
            for name, rows in (("uniform", uniform), ("market", market))
        }
    return card


def format_scorecard(card: dict[str, Any]) -> str:
    lines = [f"Forecasts: {card['n_recorded']} recorded, {card['n_unusable']} unusable, {card['n_missing']} missing"]
    for metric, value in card["metrics"].items():
        mean = value["mean"]
        lines.append(f"{metric}: {mean:.6f}" if mean is not None else f"{metric}: unavailable")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--n-rollouts", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", type=Path, help="Report this run minus the positional result file.")
    parser.add_argument("--pair-rollouts", action="store_true", help="Pair event/date/rollout for a memory comparison.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.compare:
        card = compare_runs(
            args.results, args.compare, dataset=args.dataset,
            pair_rollouts=args.pair_rollouts, bootstrap=args.bootstrap, seed=args.seed,
        )
    else:
        card = score_run(
            args.results, dataset=args.dataset, n_rollouts=args.n_rollouts,
            bootstrap=args.bootstrap, seed=args.seed,
        )
    text = json.dumps(card, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
