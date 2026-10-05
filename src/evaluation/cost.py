"""Record model usage and estimate standard API token costs."""

from __future__ import annotations

import json
import logging
import time
from itertools import count
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# USD per million input, cached-input, and output tokens.
PRICES = {
    "claude-opus-4-6": (5.0, 0.50, 25.0),
    "claude-opus-4-8": (5.0, 0.50, 25.0),
    "gpt-5.6-sol": (4.0, 0.40, 20.0),
    "gpt-5.5": (5.0, 0.50, 30.0),
    "gpt-5.4": (2.5, 0.25, 15.0),
}

_default_model_id = ""
_default_backend = ""
_cache_write_ttl = "5m"
_warned_unpriced: set[str] = set()
_call_counter = count()


def set_default_model(model_id: str, backend: str, *, cache_write_ttl: str = "5m") -> None:
    global _default_model_id, _default_backend, _cache_write_ttl
    _default_model_id = model_id
    _default_backend = backend
    _cache_write_ttl = cache_write_ttl


def trajectory_cost(
    metrics: dict[str, Any] | None, model_id: str | None = None,
) -> dict[str, Any] | None:
    """Estimate direct API costs from input totals that include cached tokens."""
    if not metrics or _default_backend not in ("anthropic", "openai"):
        return None
    model_id = model_id or _default_model_id
    prices = PRICES.get(model_id)
    if prices is None:
        if model_id not in _warned_unpriced:
            logger.warning("No token price for %s; cost fields will be null", model_id)
            _warned_unpriced.add(model_id)
        return None

    def total(key: str) -> int:
        value = metrics.get(key)
        return int(value.get("total", 0)) if isinstance(value, dict) else int(value or 0)

    p_in, p_cache, p_out = prices
    paid_writes = _default_backend == "anthropic" or model_id.startswith("gpt-5.6-sol")
    if paid_writes and metrics.get("cache_write_input_tokens") is None:
        return None
    cached = total("cache_read_input_tokens")
    written = total("cache_write_input_tokens") if paid_writes else 0
    uncached = total("input_tokens") - cached - written
    if uncached < 0:
        logger.warning("Inconsistent input/cache token totals; cost fields will be null")
        return None
    write_multiplier = (
        {"5m": 1.25, "1h": 2.0}.get(_cache_write_ttl)
        if _default_backend == "anthropic" else 1.25
    )
    if written and write_multiplier is None:
        return None
    usd_in = uncached * p_in / 1e6
    usd_cache = cached * p_cache / 1e6
    usd_write = written * p_in * (write_multiplier or 1.0) / 1e6
    usd_out = total("output_tokens") * p_out / 1e6
    return {
        "usd_input": round(usd_in, 6),
        "usd_cache_read": round(usd_cache, 6),
        "usd_cache_write": round(usd_write, 6),
        "usd_output": round(usd_out, 6),
        "usd_total": round(usd_in + usd_cache + usd_write + usd_out, 6),
        "price_per_mtok": {"input": p_in, "cache_read": p_cache, "output": p_out},
        "cache_write_ttl": _cache_write_ttl if written and _default_backend == "anthropic" else None,
    }


def _scan_event(event: Any, record: dict[str, Any]) -> None:
    if not isinstance(event, dict):
        return
    metadata = event.get("metadata")
    usage = metadata.get("usage") if isinstance(metadata, dict) else event.get("usage")
    if isinstance(usage, dict):
        for source, target in (
            ("inputTokens", "in_tok"),
            ("outputTokens", "out_tok"),
            ("totalTokens", "total_tok"),
            ("cacheReadInputTokens", "cache_read_tok"),
            ("cacheWriteInputTokens", "cache_write_tok"),
        ):
            if isinstance(usage.get(source), int):
                record[target] = record.get(target, 0) + usage[source]
        for key, value in usage.items():
            if "reasoning" in key.lower() and isinstance(value, int):
                record["reasoning_tok_exact"] = record.get("reasoning_tok_exact", 0) + value


class _TappedModel:
    """Observe native usage events without modifying SDK clients or responses."""

    def __init__(self, inner: Any, sink: Path, model_id: str, instance_idx: int):
        self._inner = inner
        self._sink = sink
        self._model_id = model_id
        self._instance = instance_idx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def stream(self, *args: Any, **kwargs: Any):
        record = {
            "model_id": self._model_id,
            "backend": _default_backend,
            "model_instance": self._instance,
            "call_idx": next(_call_counter),
            "ts": time.time(),
        }
        started = time.monotonic()
        try:
            async for event in self._inner.stream(*args, **kwargs):
                try:
                    _scan_event(event, record)
                except (TypeError, AttributeError, ValueError):
                    logger.debug("Unrecognized usage event", exc_info=True)
                yield event
        finally:
            try:
                record["latency_s"] = round(time.monotonic() - started, 4)
                # Anthropic reports uncached input separately from cache reads/writes.
                if _default_backend == "anthropic" and "in_tok" in record:
                    record["uncached_in_tok"] = record["in_tok"]
                    record["in_tok"] += record.get("cache_read_tok", 0) + record.get("cache_write_tok", 0)
                    if "out_tok" in record:
                        record["total_tok"] = record["in_tok"] + record["out_tok"]
                with self._sink.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record) + "\n")
            except Exception as error:
                logger.warning("Usage record write failed: %s", error)


def tap_factory(factory: Any, sink_path: str | Path, model_id: str) -> Any:
    sink = Path(sink_path)
    try:
        sink.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.warning("Usage logging disabled: %s", error)
        return factory
    instances = count()

    def wrapped() -> Any:
        return _TappedModel(factory(), sink, model_id, next(instances))

    return wrapped
