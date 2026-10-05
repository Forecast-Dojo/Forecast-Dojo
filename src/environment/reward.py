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

"""Per-step forecast reward — Brier and log score against the resolved outcome."""

from __future__ import annotations

import math
from typing import Literal

from strands_env.core.types import RewardResult

from .notebook import (
    AnswerParse,
    NotebookParse,
    extract_answer_detailed,
    extract_notebook_detailed,
)

Metric = Literal["brier", "log_prob"]

_WORST_BRIER = 2.0


def _p_matches(answer: dict[str, float] | None, notebook_p: dict[str, float] | None, tol: float) -> bool:
    """Whether the `<answer>` and notebook `assessment.p` distributions agree."""
    if not answer or not notebook_p:
        return False
    if set(answer) != set(notebook_p):
        return False
    return all(abs(answer[k] - notebook_p[k]) <= tol for k in answer)


def normalize_distribution(p: dict[str, float]) -> tuple[dict[str, float], float]:
    """Clamp negative entries to 0 and renormalize to sum 1."""
    clamped = {str(k): max(0.0, float(v)) for k, v in p.items()}
    raw_sum = sum(clamped.values())
    if raw_sum <= 0:
        return {}, raw_sum
    return {k: v / raw_sum for k, v in clamped.items()}, raw_sum


def brier_score(p: dict[str, float], truth: str) -> float:
    """Multi-class Brier score: ``sum_k (p_k - 1[k == truth])^2``."""
    labels = set(p) | {truth}
    return sum((p.get(k, 0.0) - (1.0 if k == truth else 0.0)) ** 2 for k in labels)


def accuracy_score(p: dict[str, float], truth: str) -> float:
    """Top-1 accuracy with fractional credit for tied labels."""
    if not p:
        return 0.0
    max_p = max(p.values())
    if max_p <= 0.0:
        return 0.0
    winners = [k for k, v in p.items() if v == max_p]
    return 1.0 / len(winners) if truth in winners else 0.0


def log_prob_score(p: dict[str, float], truth: str, eps: float) -> float:
    """Log-likelihood of the truth label: ``log(p_truth)``, clamped to `eps`."""
    p_y = p.get(truth, 0.0)
    return math.log(max(eps, min(1.0, p_y)))


class ForecastReward:
    """Outcome-based step reward with an optional format penalty."""

    def __init__(
        self,
        metric: Metric = "log_prob",
        *,
        normalize: bool = True,
        log_loss_eps: float = 1e-3,
        format_coef: float = 1.0,
        sum_tol: float = 0.05,
        score_notebook: bool = True,
    ) -> None:
        if metric not in ("brier", "log_prob"):
            raise ValueError(f"metric must be 'brier' or 'log_prob'; got {metric!r}")
        self.metric: Metric = metric
        self.normalize = normalize
        self.log_loss_eps = float(log_loss_eps)
        self.format_coef = float(format_coef)
        self.sum_tol = float(sum_tol)
        self.score_notebook = bool(score_notebook)

    def _format_quality(
        self, answer: AnswerParse, notebook: NotebookParse, prob_sum_raw: float
    ) -> dict[str, float]:
        """Score answer and notebook format quality in [0, 1]."""
        sum_ok = abs(prob_sum_raw - 1.0) <= self.sum_tol
        answer_ok = (int(answer.strict_json) + int(answer.all_numeric) + int(sum_ok)) / 3.0

        if not self.score_notebook:
            return {
                "format_quality": answer_ok,
                "format_answer_ok": answer_ok,
                "format_answer_strict_json": float(answer.strict_json),
                "format_answer_all_numeric": float(answer.all_numeric),
                "format_answer_sum_ok": float(sum_ok),
            }

        p_match = _p_matches(answer.forecast, notebook.p, self.sum_tol)
        notebook_ok = (
            int(notebook.parseable) + int(notebook.has_assessment_p) + int(p_match)
        ) / 3.0

        return {
            "format_quality": 0.5 * answer_ok + 0.5 * notebook_ok,
            "format_answer_ok": answer_ok,
            "format_notebook_ok": notebook_ok,
            "format_answer_strict_json": float(answer.strict_json),
            "format_answer_all_numeric": float(answer.all_numeric),
            "format_answer_sum_ok": float(sum_ok),
            "format_notebook_parseable": float(notebook.parseable),
            "format_notebook_has_p": float(notebook.has_assessment_p),
            "format_p_matches_answer": float(p_match),
        }

    def _select(self, brier: float, log_prob: float) -> float:
        """Map the two scores to the scalar reward for the active metric."""
        return -brier if self.metric == "brier" else log_prob

    def _result(
        self,
        *,
        brier: float,
        log_prob: float,
        accuracy: float,
        p_truth: float,
        forecast: dict[str, float],
        truth: str,
        parse_ok: bool,
        prob_sum_raw: float,
        crowd_prob: float | None,
        format_components: dict[str, float],
    ) -> RewardResult:
        """Combine outcome and format scores with their diagnostics."""
        info_alpha: float | None = None
        if crowd_prob is not None:
            crowd_log_prob = math.log(max(self.log_loss_eps, min(1.0, float(crowd_prob))))
            info_alpha = log_prob - crowd_log_prob
        proper = self._select(brier, log_prob)

        reward = proper + self.format_coef * (format_components["format_quality"] - 1.0)
        return RewardResult(
            reward=reward,
            info={
                "metric": self.metric,
                "reward": reward,
                "proper_score": proper,
                "format_penalty": reward - proper,
                "brier": brier,
                "log_prob": log_prob,
                "accuracy": accuracy,
                "p_truth": p_truth,
                "info_alpha": info_alpha,
                "crowd_prob": crowd_prob,
                "forecast": forecast,
                "truth": truth,
                "parse_ok": parse_ok,
                "prob_sum_raw": prob_sum_raw,
                "truth_in_forecast": truth in (forecast or {}),
                **format_components,
            },
        )

    async def score_step(
        self, *, notebook: str | None, label: str | None, crowd_prob: float | None = None
    ) -> RewardResult:
        """Score one step's response against the resolved label."""
        truth = (label or "").strip()
        if not truth or truth == "?":
            return RewardResult(reward=0.0, info={"error": "missing_truth", "metric": self.metric})

        worst_brier = _WORST_BRIER
        worst_log_prob = math.log(self.log_loss_eps)

        answer = extract_answer_detailed(notebook)
        nb = extract_notebook_detailed(notebook)
        forecast = answer.forecast

        if not forecast:
            fmt = self._format_quality(answer, nb, prob_sum_raw=0.0)
            return self._result(
                brier=worst_brier, log_prob=worst_log_prob, accuracy=0.0, p_truth=0.0,
                forecast={}, truth=truth, parse_ok=False, prob_sum_raw=0.0, crowd_prob=crowd_prob,
                format_components=fmt,
            )

        if self.normalize:
            dist, raw_sum = normalize_distribution(forecast)
        else:
            dist = {str(k): max(0.0, float(v)) for k, v in forecast.items()}
            raw_sum = sum(dist.values())

        fmt = self._format_quality(answer, nb, prob_sum_raw=raw_sum)

        if not dist or raw_sum <= 0:
            return self._result(
                brier=worst_brier, log_prob=worst_log_prob, accuracy=0.0, p_truth=0.0,
                forecast={}, truth=truth, parse_ok=True, prob_sum_raw=raw_sum, crowd_prob=crowd_prob,
                format_components=fmt,
            )

        brier = brier_score(dist, truth)
        log_prob = log_prob_score(dist, truth, self.log_loss_eps)
        accuracy = accuracy_score(dist, truth)
        return self._result(
            brier=brier, log_prob=log_prob, accuracy=accuracy, p_truth=dist.get(truth, 0.0),
            forecast=dist, truth=truth, parse_ok=True, prob_sum_raw=raw_sum, crowd_prob=crowd_prob,
            format_components=fmt,
        )
