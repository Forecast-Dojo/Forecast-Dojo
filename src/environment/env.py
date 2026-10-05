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

"""RAG-style deep-research environment backed by a local FAISS corpus."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from strands import Agent
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeModelCallEvent
from strands.types.content import Messages
from strands_sglang import Rollout, ToolLimiter
from typing_extensions import Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import Observation, RewardFunction, TerminationReason

from prompts import load_prompt, prompt_filename, prompt_path

from .code_tool import LocalPythonToolkit
from .notebook import extract_belief_notebook
from .tool import RetrieverToolkit, set_window


class ForecastMode(str, Enum):
    """Forecast modes and their metadata layout."""

    MEMORY_ON = "memory-on"
    MEMORY_FREE = "memory-free"

    @property
    def uses_notebook(self) -> bool:
        """Whether the belief notebook is in the prompt, carried, and scored."""
        return self is ForecastMode.MEMORY_ON

    def system_prompt_filename(self, enable_code: bool = True) -> str:
        return prompt_filename(self.value, enable_code=enable_code)

    def parse_metadata(self, meta: dict[str, Any]) -> tuple[list[str], list[float | None]]:
        """Read this mode's data-row shape into `(forecast_dates, crowd_probs)`."""
        if self is ForecastMode.MEMORY_FREE:
            return [meta["forecast_date"]], [meta.get("crowd_prob")]
        return list(meta["forecast_dates"]), list(meta.get("crowd_probs") or [])


class ForecastStep(BaseModel):
    """One forecast date's trajectory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    forecast_date: str
    final_response: str | None = None
    messages: Messages = Field(default_factory=list)
    termination_reason: TerminationReason = TerminationReason.NOT_TERMINATED
    metrics: dict[str, Any] = Field(default_factory=dict)
    rollout: Rollout | None = None


class EpisodeResult(BaseModel):
    """Per-date trajectories and aggregate episode metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    steps: list[ForecastStep] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    termination_reason: TerminationReason = TerminationReason.NOT_TERMINATED

    @property
    def messages(self) -> Messages:
        """All step messages concatenated, in conversation order."""
        return [m for s in self.steps for m in s.messages]


class ForecastConfig(EnvironmentConfig, total=False):
    """Serializable configuration for `ForecastEnv`."""

    mode: str
    embedding_endpoint: str
    embedding_model: str
    index_root: str
    text_token_budget: int
    date_filter: list[str] | None
    start_date: str | None
    cutoff_date: str | None
    retriever_timeout: int

    decay_alpha: float
    decay_horizon_days: int
    rerank_oversample: int

    enable_code: bool

    no_tools: bool
    code_execution_timeout: int
    code_concurrency: int
    code_memory_limit_mb: int

    code_session_timeout_seconds: int

SYNTHESIS_DIRECTIVE = (
    "Your retrieval budget is exhausted — do not call any more tools. Using only the "
    "evidence you have already retrieved, output your belief notebook and <answer> now."
)


class ContextSynthesisHook(HookProvider):
    """Force a final synthesis turn before the context window runs out."""

    def __init__(self, context_len: int, ratio: float) -> None:
        self.budget = int(context_len * ratio) if ratio < 1.0 else None
        self.fired = False

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._maybe_synthesize)

    def _maybe_synthesize(self, event: BeforeModelCallEvent) -> None:
        if self.budget is None or self.fired:
            return
        if (event.projected_input_tokens or 0) <= self.budget:
            return
        self.fired = True
        event.agent.messages.append({"role": "user", "content": [{"text": SYNTHESIS_DIRECTIVE}]})


class ForecastEnv(Environment):
    """Deep-research environment over a local FAISS news corpus."""

    default_system_prompt_path = prompt_path("memory-on")

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        retriever_concurrency: asyncio.Semaphore | int = 10,
        code_client: object | None = None,
        code_quotas: object | None = None,
        **config: Unpack[ForecastConfig],
    ):
        super().__init__(model_factory=model_factory, reward_fn=reward_fn, **config)  # type: ignore[misc]

        self.mode = ForecastMode(self.config.get("mode", "memory-on"))

        self.no_tools: bool = bool(self.config.get("no_tools", False))
        if self.no_tools:
            self.config["enable_code"] = False  # type: ignore[typeddict-unknown-key]
        self.retriever_toolkit = None if self.no_tools else RetrieverToolkit(
            embedding_endpoint=self.config["embedding_endpoint"],  # type: ignore[typeddict-item]
            index_root=self.config["index_root"],  # type: ignore[typeddict-item]
            embedding_model=self.config.get("embedding_model", "Qwen3-Embedding-8B"),  # type: ignore[arg-type]
            text_token_budget=int(self.config.get("text_token_budget", 8000)),
            date_filter=self.config.get("date_filter"),  # type: ignore[arg-type]
            start_date=self.config.get("start_date"),  # type: ignore[arg-type]
            cutoff_date=self.config.get("cutoff_date"),  # type: ignore[arg-type]
            timeout=int(self.config.get("retriever_timeout", 30)),
            concurrency=retriever_concurrency,
            decay_alpha=float(self.config.get("decay_alpha", 1.0)),
            decay_horizon_days=int(self.config.get("decay_horizon_days", 180)),
            rerank_oversample=int(self.config.get("rerank_oversample", 1)),
        )

        self.search_tool = None if self.no_tools else self.retriever_toolkit.search
        self.scrape_tool = None if self.no_tools else self.retriever_toolkit.scrape

        self.enable_code: bool = bool(self.config.get("enable_code", True))

        if not self.config.get("system_prompt"):
            self.system_prompt = load_prompt(
                self.mode.value, search=not self.no_tools, enable_code=self.enable_code
            )

        self.code_toolkit: LocalPythonToolkit | None = None
        if self.enable_code:
            self.code_toolkit = LocalPythonToolkit(
                execution_timeout=int(
                    self.config.get(
                        "code_execution_timeout",
                        LocalPythonToolkit.DEFAULT_EXECUTION_TIMEOUT_SECONDS,
                    )
                ),
                concurrency=int(self.config.get("code_concurrency", 24)),
                memory_limit_mb=int(self.config.get("code_memory_limit_mb", 4096)),
            )

    @override
    def get_hooks(self) -> list:
        """Base hooks plus context-synthesis, when enabled."""
        hooks = list(super().get_hooks())
        ratio = float(self.config.get("context_synth_ratio", os.getenv("FORECAST_CONTEXT_SYNTH_RATIO", "1.0")))
        if ratio < 1.0:
            hooks.append(ContextSynthesisHook(
                int(self.config.get("context_len", os.getenv("FORECAST_CONTEXT_LEN", "131072"))), ratio))
        return hooks

    @override
    def get_tools(self) -> list:
        """Return the forecasting toolset: search + scrape, plus code if enabled."""
        if self.no_tools:
            return []
        tools = [self.search_tool, self.scrape_tool]
        if self.code_toolkit is not None:
            tools.append(self.code_toolkit.python)
        return tools

    async def run_episode(
        self,
        question_prompt: str,
        forecast_dates: Sequence[str],
        start_date: str | None = None,
    ) -> EpisodeResult:
        """Run one forecast episode over `forecast_dates`; one `ForecastStep` per date."""
        if not forecast_dates:
            raise ValueError("run_episode requires at least one forecast date")

        use_notebook = self.mode.uses_notebook

        model = self.model_factory()

        steps: list[ForecastStep] = []
        episode_reason = TerminationReason.TASK_COMPLETE
        tool_iters = tool_calls = cancelled_tool_calls = 0
        prev_notebook: str | None = None

        per_step_compute: list[dict[str, Any]] = []

        episode_tool_parse_errors: dict[str, int] = {}

        for fdate in forecast_dates:
            step, step_compute = await self._run_step(
                model, question_prompt, fdate, start_date, prev_notebook,
                include_notebook=use_notebook,
            )
            tool_iters += step.metrics["tool_iters"]
            tool_calls += step.metrics["tool_calls"]
            cancelled_tool_calls += step.metrics["cancelled_tool_calls"]
            per_step_compute.append(step_compute)

            for tool_name, count in (getattr(model, "tool_parse_errors", None) or {}).items():
                episode_tool_parse_errors[tool_name] = (
                    episode_tool_parse_errors.get(tool_name, 0) + count
                )
            steps.append(step)
            if step.termination_reason is not TerminationReason.TASK_COMPLETE:
                episode_reason = step.termination_reason
                break

            if hasattr(model, "reset"):
                model.reset()

            if use_notebook:
                prev_notebook = extract_belief_notebook(step.final_response)

        compute_summary = self._aggregate_compute_metrics(
            per_step_compute, tool_parse_errors=episode_tool_parse_errors,
        )
        metrics = {
            "n_steps": len(steps),
            "tool_iters": tool_iters,
            "tool_calls": tool_calls,
            "cancelled_tool_calls": cancelled_tool_calls,
            **compute_summary,
        }
        return EpisodeResult(
            steps=steps,
            metrics=metrics,
            termination_reason=episode_reason,
        )

    async def _run_step(
        self,
        model,
        question_prompt: str,
        forecast_date: str,
        start_date: str | None,
        prev_notebook: str | None,
        *,
        include_notebook: bool,
    ) -> tuple[ForecastStep, dict[str, Any]]:
        """Run one forecast at `forecast_date` and snapshot its standalone step."""
        set_window(start_date, forecast_date)

        tool_limiter = ToolLimiter(
            max_tool_iters=self.max_tool_iters,
            max_tool_calls=self.max_tool_calls,
            max_parallel_tool_calls=self.max_parallel_tool_calls,
        )
        agent = Agent(
            model=model,
            tools=list(self.get_tools()),
            system_prompt=self.system_prompt,
            hooks=[tool_limiter] + list(self.get_hooks()),
            conversation_manager=self.get_conversation_manager(),
            callback_handler=PrintingCallbackHandler() if self.verbose else None,
            trace_attributes=self.trace_attributes or None,
            name=self.agent_name,
        )

        message = self._build_step_prompt(
            question_prompt, forecast_date, prev_notebook, include_notebook=include_notebook
        )
        error: Exception | None = None
        try:
            await agent.invoke_async(message)
        except Exception as e:
            error = e
        reason = TerminationReason.from_error(error)

        step_compute = self.compute_metrics(agent.event_loop_metrics, tool_parse_errors=None)
        step_metrics = {
            "tool_iters": tool_limiter.tool_iter_count,
            "tool_calls": tool_limiter.tool_call_count,
            "cancelled_tool_calls": tool_limiter.cancelled_tool_call_count,
            **step_compute,
        }
        step_rollout = getattr(model, "rollout", None)
        step_messages = list(agent.messages)
        step = ForecastStep(
            forecast_date=forecast_date,
            final_response=self._step_final_response(step_messages),
            messages=step_messages,
            termination_reason=reason,
            metrics=step_metrics,
            rollout=step_rollout,
        )
        return step, step_compute

    @staticmethod
    def render_question(q: dict[str, Any]) -> str:
        """Render the static question content shown on every step of an episode."""
        title = q["title"]
        body = (q.get("body") or "").strip()

        res_date = (q.get("close_date") or "")[:10]
        kind = q["belief_kind"]

        if kind == "binary":
            outcomes = (
                "Outcomes: YES / NO. "
                "Forecast a probability for each label; the two probabilities must sum to 1."
            )
        else:
            legs = "\n".join(
                f"  - {m['label']!r}: {m.get('question') or '(no per-leg question)'}"
                for m in q.get("markets") or []
            )
            outcomes = (
                f"Outcomes ({len(q.get('markets') or [])}, exactly one resolves YES):\n"
                f"{legs}\n"
                "Forecast a probability for each label; the K probabilities must sum to 1."
            )

        parts = [f"Question: {title}", f"Resolves: {res_date}", outcomes]
        if body:
            parts.append(f"\nResolution criteria / context:\n{body}")
        return "\n".join(parts)

    @staticmethod
    def _build_step_prompt(
        question_prompt: str,
        forecast_date: str,
        prev_notebook: str | None,
        *,
        include_notebook: bool = True,
    ) -> str:
        """Append the current date and optional notebook to the fixed question."""
        prompt = f"{question_prompt}\n\nForecast date: {forecast_date}"
        if include_notebook:
            prompt += f"\n\nBelief Notebook: {prev_notebook if prev_notebook is not None else 'None'}"
        return prompt

    @staticmethod
    def _step_final_response(messages: Messages) -> str | None:
        """Return the final assistant text with reasoning tags removed."""
        return Observation(messages=messages).final_response

    @staticmethod
    def _aggregate_compute_metrics(
        per_step: list[dict[str, Any]],
        tool_parse_errors: dict[str, int],
    ) -> dict[str, Any]:
        """Aggregate per-step compute metrics."""
        if not per_step:
            return {}

        def _merge_summary(blocks: list[dict[str, int | float] | None]) -> dict[str, int | float] | None:
            """Combine step-level `_summarize` blocks. Returns None if all are None."""
            present = [b for b in blocks if b]
            if not present:
                return None
            total = sum(b["total"] for b in present)
            mx = max(b["max"] for b in present)
            mn = min(b["min"] for b in present)

            n = sum(b["total"] / b["mean"] for b in present if b["mean"]) or 1
            return {"total": total, "max": mx, "mean": round(total / n, 4), "min": mn}

        model_calls = sum(r.get("model_calls", 0) for r in per_step)

        model_latency_s = _merge_summary([r.get("model_latency_s") for r in per_step])
        input_tokens = _merge_summary([r.get("input_tokens") for r in per_step])
        output_tokens = _merge_summary([r.get("output_tokens") for r in per_step])
        cache_read_input_tokens = _merge_summary([r.get("cache_read_input_tokens") for r in per_step])
        cache_write_input_tokens = _merge_summary([r.get("cache_write_input_tokens") for r in per_step])
        if any(r.get("model_calls", 0) and r.get("cache_write_input_tokens") is None for r in per_step):
            cache_write_input_tokens = None

        cache_hit_rate: float | None = None
        if input_tokens and cache_read_input_tokens and input_tokens["total"] > 0:
            cache_hit_rate = round(cache_read_input_tokens["total"] / input_tokens["total"], 4)

        per_tool: dict[str, dict[str, Any]] = {}
        for r in per_step:
            for name, tm in (r.get("per_tool_metrics") or {}).items():
                acc = per_tool.setdefault(name, {
                    "is_known": tm["is_known"], "calls": 0, "successes": 0,
                    "errors": 0, "parse_errors": 0, "latency_s": 0.0,
                })
                acc["calls"] += tm["calls"]
                acc["successes"] += tm["successes"]
                acc["errors"] += tm["errors"]
                acc["latency_s"] = round(acc["latency_s"] + tm["latency_s"], 4)
        for name, count in tool_parse_errors.items():
            per_tool.setdefault(name, {
                "is_known": False, "calls": 0, "successes": 0,
                "errors": 0, "parse_errors": 0, "latency_s": 0.0,
            })["parse_errors"] = count

        return {
            "model_calls": model_calls,
            "model_latency_s": model_latency_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_write_input_tokens": cache_write_input_tokens,
            "cache_hit_rate": cache_hit_rate,
            "per_tool_metrics": per_tool or None,
        }

    async def cleanup(self) -> None:
        """Close the retriever's HTTP client."""
        if self.retriever_toolkit is not None:
            await self.retriever_toolkit.cleanup()
