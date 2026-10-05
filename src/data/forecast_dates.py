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

"""Select forecast dates using market movement and corpus news volume."""

from __future__ import annotations

import math
from datetime import date

# Minimum distance from recorded close; content leakage is checked separately.
DEFAULT_MIN_DAYS_BEFORE_RESOLUTION = 2


def _belief_delta_series(q: dict) -> list[float]:
    """Per-day total-variation belief change vs. the previous day (index 0 = 0)."""
    n = len(q["daily_index"])
    if q["belief_kind"] == "binary":
        series = next(iter(q["raw_yes_history"].values()))
        out = [0.0]
        for t in range(1, n):
            a, b = series[t], series[t - 1]
            out.append(abs(a - b) if (a is not None and b is not None) else 0.0)
        return out

    hist = q["normalized_history"]
    labels = list(hist.keys())
    out = [0.0]
    for t in range(1, n):
        col_t = [hist[k][t] for k in labels]
        col_p = [hist[k][t - 1] for k in labels]
        if any(v is None for v in col_t) or any(v is None for v in col_p):
            out.append(0.0)
            continue
        out.append(0.5 * sum(abs(a - b) for a, b in zip(col_t, col_p)))
    return out


def crowd_prob_on_truth(q: dict) -> list[float | None]:
    """Return daily market probability on the realized outcome; gaps remain None."""
    di = q.get("daily_index") or []
    n = len(di)
    kind = q.get("belief_kind")

    if kind == "binary":
        yes_hist = q.get("raw_yes_history") or {}
        if not yes_hist:
            return [None] * n
        yes_vals = next(iter(yes_hist.values()))
        resolved = (q.get("resolved_label") or "").lower()
        if resolved == "yes":
            return [None if v is None else float(v) for v in yes_vals]
        if resolved == "no":
            return [None if v is None else 1.0 - float(v) for v in yes_vals]
        return [None] * n

    if kind == "multi_neg_risk":
        norm = q.get("normalized_history") or {}
        labels = list(norm.keys())
        win_idx = q.get("winner_market_index")
        if win_idx is None or not (0 <= win_idx < len(labels)):
            return [None] * n
        return [None if v is None else float(v) for v in norm[labels[win_idx]]]

    return [None] * n


def crowd_probs_for_dates(q: dict, dates: list[str]) -> list[float | None]:
    """Return outcome probabilities aligned to the requested YYYY-MM-DD dates."""
    series = crowd_prob_on_truth(q)
    by_date = {d[:10]: p for d, p in zip(q.get("daily_index") or [], series)}
    return [by_date.get(d[:10]) for d in dates]


def _normalized(series: list[float]) -> list[float]:
    """Scale a non-negative series to [0, 1] by its own peak (0 if flat)."""
    peak = max(series) if series else 0.0
    if peak <= 0:
        return [0.0] * len(series)
    return [x / peak for x in series]


def _belief_score_series(q: dict) -> list[float]:
    """Belief delta normalized to [0, 1] by the question's own max delta."""
    return _normalized(_belief_delta_series(q))


def _volume_score_series(q: dict) -> list[float]:
    """Max-normalize daily corpus article counts; missing or misaligned input yields zeros."""
    n = len(q["daily_index"])
    news = q.get("news_daily")
    if not news or len(news) != n:
        return [0.0] * n
    return _normalized([float(x or 0.0) for x in news])


def _resolution_date(q: dict) -> date | None:
    """Best-effort resolution date: close_date, else latest leg closed_time, else legacy fields."""
    cd = q.get("close_date") or ""
    if cd:
        try:
            return date.fromisoformat(cd[:10])
        except ValueError:
            pass
    closes: list[date] = []
    for m in q.get("markets") or []:
        ct = m.get("closed_time") or ""
        if ct:
            try:
                closes.append(date.fromisoformat(ct[:10]))
            except ValueError:
                pass
    if closes:
        return max(closes)
    for field in ("resolution_date", "end_date"):
        v = q.get(field) or ""
        if v:
            try:
                return date.fromisoformat(v[:10])
            except ValueError:
                pass
    return None


def _trim_for_leakage(dates: list[str], q: dict, *, min_days_before_resolution: int) -> list[str]:
    """Drop candidates within the leakage buffer of resolution."""
    if min_days_before_resolution <= 0:
        return dates
    res_d = _resolution_date(q)
    if res_d is None:
        return dates
    eligible = [
        d for d in dates
        if (res_d - date.fromisoformat(d)).days >= min_days_before_resolution
    ]
    return eligible


def compute_k(horizon_days: int, *, k_min: int = 3, k_max: int = 10, c: float = 1.0) -> int:
    """Sub-linear K schedule: K(H) = clamp(round(c * sqrt(H)), k_min, k_max)."""
    if k_min < 1 or k_max < k_min:
        raise ValueError(f"need 1 <= k_min <= k_max; got ({k_min}, {k_max})")
    if horizon_days < 1:
        return k_min
    return max(k_min, min(k_max, round(c * math.sqrt(horizon_days))))


def pick_forecast_dates(
    q: dict,
    k: int | None = None,
    *,
    alpha: float = 0.7,
    eps: float = 1e-3,
    auto_k: bool = False,
    k_min: int = 3,
    k_max: int = 10,
    c: float = 1.0,
    min_days_before_resolution: int = DEFAULT_MIN_DAYS_BEFORE_RESOLUTION,
) -> list[str]:
    """Select chronological dates using fixed K or a sublinear schedule."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if auto_k and k is not None:
        raise ValueError("pass either `k` or `auto_k=True`, not both")
    if not auto_k and k is None:
        raise ValueError("must pass `k` or set `auto_k=True`")
    if min_days_before_resolution < 0:
        raise ValueError(f"min_days_before_resolution must be >= 0; got {min_days_before_resolution}")

    all_dates = [d[:10] for d in q["daily_index"]]
    if not all_dates:
        return []

    # Size K using only eligible dates.
    dates = _trim_for_leakage(all_dates, q, min_days_before_resolution=min_days_before_resolution)
    n = len(dates)
    if n == 0:
        return []

    k_resolved = compute_k(n, k_min=k_min, k_max=k_max, c=c) if auto_k else int(k)
    if k_resolved < 1:
        raise ValueError(f"resolved k must be >= 1; got {k_resolved}")
    if n <= k_resolved:
        return dates
    k = k_resolved

    # Align full-history signals to the eligible dates.
    full_idx = {d: i for i, d in enumerate(all_dates)}
    eligible_orig_idx = [full_idx[d] for d in dates]
    belief_full = _belief_score_series(q)
    volume_full = _volume_score_series(q)
    belief = [belief_full[i] for i in eligible_orig_idx]
    volume = [volume_full[i] for i in eligible_orig_idx]
    weights = [
        alpha * belief[t] + (1.0 - alpha) * volume[t] + eps * (t / max(n - 1, 1))
        for t in range(n)
    ]

    # Choose one maximum-scoring date per equal-count temporal bin.
    picks: list[int] = []
    for i in range(k):
        lo = math.floor(n * i / k)
        hi = max(math.floor(n * (i + 1) / k), lo + 1)
        best = lo
        for t in range(lo + 1, hi):
            if weights[t] >= weights[best]:  # Break ties toward the later date.
                best = t
        picks.append(best)

    return [dates[t] for t in picks]
