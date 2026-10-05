"""Validate frozen forecast schedules and export event or date-level tasks."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcome_labels(event):
    if event["belief_kind"] == "binary":
        return ["YES", "NO"]
    return list(dict.fromkeys(market["label"] for market in event["metadata"]["question"]["markets"]))


def validate(events, min_dates=3):
    seen = set()
    kinds = Counter()
    dates = []
    for event in events:
        event_id = str(event["event_id"])
        if event_id in seen:
            raise ValueError(f"Duplicate event: {event_id}")
        seen.add(event_id)
        metadata = event["metadata"]
        schedule = metadata["forecast_dates"]
        if len(schedule) < min_dates or schedule != sorted(set(schedule)):
            raise ValueError(f"Invalid forecast schedule: {event_id}")
        probabilities = metadata["crowd_probs"]
        if len(probabilities) != len(schedule):
            raise ValueError(f"Unaligned crowd probabilities: {event_id}")
        for probability in probabilities:
            if probability is not None and (
                not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError(f"Invalid crowd probability: {event_id}")
        labels = outcome_labels(event)
        if len(labels) < 2 or event["label"] not in labels:
            raise ValueError(f"Invalid outcome labels: {event_id}")
        start = dt.date.fromisoformat(metadata["start_date"])
        close = dt.date.fromisoformat(metadata["question"]["close_date"])
        for date in schedule:
            day = dt.date.fromisoformat(date)
            if day < start or (close - day).days < 2:
                raise ValueError(f"Forecast outside the permitted window: {event_id}/{date}")
        distributions = metadata.get("crowd_distributions")
        if distributions is not None:
            if len(distributions) != len(schedule):
                raise ValueError(f"Unaligned crowd distributions: {event_id}")
            for distribution in distributions:
                if distribution is None:
                    continue
                if set(distribution) != set(labels) or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    for value in distribution.values()
                ) or not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-9):
                    raise ValueError(f"Invalid crowd distribution: {event_id}")
        kinds[event["belief_kind"]] += 1
        dates.extend(schedule)
    return {
        "events": len(events),
        "forecast_steps": len(dates),
        "binary_events": kinds["binary"],
        "multi_option_events": len(events) - kinds["binary"],
        "forecast_date_range": [min(dates), max(dates)] if dates else [],
    }


def filter_events(events, exclusions=(), min_dates=3):
    excluded = {(str(event_id), date) for event_id, date in exclusions}
    result = []
    for original in events:
        event = copy.deepcopy(original)
        metadata = event["metadata"]
        keep = [
            i for i, date in enumerate(metadata["forecast_dates"])
            if (str(event["event_id"]), date) not in excluded
        ]
        if len(keep) < min_dates:
            continue
        for key in ("forecast_dates", "crowd_probs", "crowd_distributions"):
            if key in metadata:
                metadata[key] = [metadata[key][i] for i in keep]
        result.append(event)
    return result


def market_distribution(event, question, date):
    index = {day[:10]: i for i, day in enumerate(question["daily_index"])}.get(date)
    if index is None:
        return None

    def value(history, label):
        series = history.get(label, [])
        return series[index] if index < len(series) else None

    if event["belief_kind"] == "binary":
        history = question.get("raw_yes_history") or {}
        if len(history) != 1:
            return None
        yes = value(history, next(iter(history)))
        return {"YES": float(yes), "NO": 1.0 - float(yes)} if yes is not None else None

    history = question.get("normalized_history") or {}
    distribution = {
        label: float(value(history, label) or 0.0) for label in outcome_labels(event)
    }
    total = sum(distribution.values())
    return {label: probability / total for label, probability in distribution.items()} if total > 0 else None


def attach_market_distributions(events, questions_path):
    questions = {str(question["event_id"]): question for question in read_jsonl(questions_path)}
    for event in events:
        question = questions[str(event["event_id"])]
        event["metadata"]["crowd_distributions"] = [
            market_distribution(event, question, date)
            for date in event["metadata"]["forecast_dates"]
        ]


def memory_free_rows(events):
    for event in events:
        metadata = event["metadata"]
        for index, date in enumerate(metadata["forecast_dates"]):
            row = copy.deepcopy(event)
            row_metadata = row["metadata"]
            row_metadata["forecast_date"] = date
            row_metadata["crowd_prob"] = metadata["crowd_probs"][index]
            del row_metadata["forecast_dates"], row_metadata["crowd_probs"]
            if "crowd_distributions" in row_metadata:
                row_metadata["crowd_distribution"] = row_metadata.pop("crowd_distributions")[index]
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Event-level JSONL with frozen forecast_dates.")
    parser.add_argument("--output", type=Path, help="Omit to validate without writing a file.")
    parser.add_argument("--mode", choices=("memory-free", "memory-on"), default="memory-on")
    parser.add_argument("--exclude", type=Path, help="JSON list of [event_id, forecast_date] pairs to remove.")
    parser.add_argument("--questions", type=Path, help="Question histories used to attach market distributions.")
    parser.add_argument("--min-dates", type=int, default=3)
    args = parser.parse_args()
    if args.min_dates < 1:
        parser.error("--min-dates must be positive")
    events = read_jsonl(args.input)
    validate(events, min_dates=1)
    if args.exclude:
        exclusions = json.loads(args.exclude.read_text())
        events = filter_events(events, exclusions, args.min_dates)
    if args.questions:
        attach_market_distributions(events, args.questions)
    summary = validate(events, args.min_dates)
    if args.output:
        if args.output.resolve() == args.input.resolve():
            parser.error("--output must differ from --input")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rows = memory_free_rows(events) if args.mode == "memory-free" else events
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
