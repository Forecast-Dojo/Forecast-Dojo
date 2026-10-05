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

"""Factories that create fresh Strands models for each environment rollout."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from strands.models import Model
from strands.types.content import Messages
from strands_sglang import SGLangClient, SGLangModel, get_client, get_tokenizer
from strands_sglang.tool_parsers import HermesToolParser, ToolParser, get_tool_parser
from transformers import PreTrainedTokenizerBase

ModelFactory = Callable[[], Model]

DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 16384}

def sglang_model_factory(
    *,
    client: SGLangClient,
    tokenizer: PreTrainedTokenizerBase,
    tool_parser: ToolParser | None = None,
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
    return_logprob: bool = True,
    return_routed_experts: bool = False,
    enable_thinking: bool = True,
) -> ModelFactory:
    """Create SGLang models with a shared client, tokenizer, and tool parser."""
    if tool_parser is None:
        tool_parser = HermesToolParser()

    return lambda: SGLangModel(
        client=client,
        tokenizer=tokenizer,
        tool_parser=tool_parser,
        sampling_params=sampling_params,
        return_logprob=return_logprob,
        return_routed_experts=return_routed_experts,
        enable_thinking=enable_thinking,
    )

def check_server_health(base_url: str, timeout: float = 5.0) -> None:
    """Check SGLang health synchronously; raise on connection failure."""
    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise ConnectionError(f"SGLang server at {base_url} is not reachable: {e}") from e

def get_model_id(base_url: str, timeout: float = 5.0) -> str:
    """Get the model's HF identifier from the SGLang server via `/get_model_info`."""
    response = httpx.get(f"{base_url}/get_model_info", timeout=timeout)
    response.raise_for_status()
    info = response.json()
    model_id = info.get("tokenizer_path") or info.get("model_path")
    if not model_id:
        raise RuntimeError(f"No tokenizer_path/model_path at {base_url}/get_model_info")
    return str(model_id)

DEFAULT_CLIENT_ARGS = {
    "timeout": httpx.Timeout(timeout=600.0, connect=5.0),
    "max_retries": 5,
}


def _sampling_params(params: dict[str, Any] | None, token_key: str) -> dict[str, Any]:
    params = dict(DEFAULT_SAMPLING_PARAMS if params is None else params)
    if "max_new_tokens" in params:
        params[token_key] = params.pop("max_new_tokens")
    return params


def anthropic_model_factory(
    *,
    model_id: str,
    sampling_params: dict[str, Any] | None = None,
    client_args: dict[str, Any] | None = None,
) -> ModelFactory:
    """Create Claude models using the official Anthropic Messages API."""
    from strands.models.anthropic import AnthropicModel

    params = _sampling_params(sampling_params, "max_tokens")
    max_tokens = params.pop("max_tokens", DEFAULT_SAMPLING_PARAMS["max_new_tokens"])
    client_args = {**DEFAULT_CLIENT_ARGS, **(client_args or {})}
    return lambda: AnthropicModel(
        model_id=model_id,
        max_tokens=max_tokens,
        params=dict(params),
        client_args=dict(client_args),
    )


def openai_model_factory(
    *,
    model_id: str,
    sampling_params: dict[str, Any] | None = None,
    client_args: dict[str, Any] | None = None,
) -> ModelFactory:
    """Create official OpenAI Chat Completions models with local message history."""
    from strands.models.openai import OpenAIModel

    params = _sampling_params(sampling_params, "max_completion_tokens")
    client_args = {**DEFAULT_CLIENT_ARGS, **(client_args or {})}
    return lambda: OpenAIModel(
        model_id=model_id,
        params=dict(params),
        client_args=dict(client_args),
    )


def openai_compatible_model_factory(
    *,
    model_id: str,
    sampling_params: dict[str, Any] | None = None,
    client_args: dict[str, Any] | None = None,
) -> ModelFactory:
    """Create Chat Completions models for local or compatible endpoints."""
    from strands.models.openai import OpenAIModel

    class ReasoningOpenAIModel(OpenAIModel):
        """Retain vendor reasoning fields across tool calls."""

        @classmethod
        def _format_regular_messages(cls, messages: Messages, **kwargs: Any) -> list[dict[str, Any]]:
            formatted = []
            for message in messages:
                content = message["content"]
                plain = {**message, "content": [block for block in content if "reasoningContent" not in block]}
                parts = super()._format_regular_messages([plain], **kwargs)
                if message["role"] == "assistant":
                    reasoning = [
                        block["reasoningContent"]["reasoningText"].get("text", "")
                        for block in content
                        if "reasoningText" in block.get("reasoningContent", {})
                    ]
                    if reasoning:
                        assistant = next((part for part in parts if part["role"] == "assistant"), None)
                        if assistant is None:
                            assistant = {"role": "assistant"}
                            parts.insert(0, assistant)
                        assistant["reasoning_content"] = "".join(reasoning)
                        assistant.setdefault("content", "")
                formatted.extend(parts)
            return formatted

    params = _sampling_params(sampling_params, "max_tokens")
    client_args = {
        **DEFAULT_CLIENT_ARGS,
        "api_key": "EMPTY",
        "base_url": "http://localhost:30000/v1",
        **(client_args or {}),
    }
    return lambda: ReasoningOpenAIModel(
        model_id=model_id,
        params=dict(params),
        client_args=dict(client_args),
    )


@dataclass
class ModelConfig:
    """Serializable model configuration."""

    backend: Literal["sglang", "anthropic", "openai", "openai-compatible"] = "sglang"

    base_url: str | None = None
    tokenizer_path: str | None = None
    tool_parser: str | None = None
    max_connections: int = 1000

    model_id: str | None = None
    client_args: dict[str, Any] = field(default_factory=dict)

    sampling_params: dict[str, Any] = field(default_factory=lambda: {"max_new_tokens": 16384})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

def build_model_factory(config: ModelConfig | dict[str, Any]) -> ModelFactory:
    """Build a model factory from a dataclass or serialized configuration."""
    if isinstance(config, dict):
        config = ModelConfig(**config)

    if config.backend == "sglang":
        base_url = config.base_url or "http://localhost:30000"
        check_server_health(base_url)
        client = get_client(base_url, max_connections=config.max_connections)
        model_id = config.model_id or get_model_id(base_url)
        return sglang_model_factory(
            client=client,
            tokenizer=get_tokenizer(config.tokenizer_path or model_id),
            tool_parser=get_tool_parser(config.tool_parser or "hermes"),
            sampling_params=config.sampling_params,
        )

    factories = {
        "anthropic": anthropic_model_factory,
        "openai": openai_model_factory,
        "openai-compatible": openai_compatible_model_factory,
    }
    if config.backend not in factories:
        raise ValueError(f"Unsupported backend for ModelConfig: {config.backend!r}")
    if not config.model_id:
        raise ValueError(f"model_id is required for {config.backend!r}")
    client_args = dict(config.client_args)
    if config.base_url:
        client_args["base_url"] = config.base_url
    return factories[config.backend](
        model_id=config.model_id,
        sampling_params=config.sampling_params,
        client_args=client_args,
    )
