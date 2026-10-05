"""Claude API client for evidence screening."""

from __future__ import annotations

from typing import Literal

from anthropic import Anthropic


class AnthropicLLMClient:
    """Share one SDK client across workers; authenticate with ANTHROPIC_API_KEY."""

    def __init__(
        self,
        model_id: str,
        max_tokens: int = 4096,
        enable_thinking: bool = True,
        thinking_schema: Literal["adaptive", "legacy"] = "adaptive",
        thinking_effort: Literal["low", "medium", "high", "max"] = "low",
        thinking_budget: int = 4000,
        max_retries: int = 10,
        timeout: float = 120.0,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_schema = thinking_schema
        self.thinking_effort = thinking_effort
        self.thinking_budget = thinking_budget
        self._client = Anthropic(max_retries=max_retries, timeout=timeout)

    def chat(self, prompt: str, system_prompt: str | None = None) -> str:
        request = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            request["system"] = system_prompt
        if not self.enable_thinking:
            request["thinking"] = {"type": "disabled"}
        elif self.thinking_schema == "adaptive":
            request["thinking"] = {"type": "adaptive"}
            request["output_config"] = {"effort": self.thinking_effort}
        else:
            request["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        with self._client.messages.stream(**request) as stream:
            response = stream.get_final_message()
        texts = [block.text for block in response.content if block.type == "text"]
        if not texts:
            raise RuntimeError("Claude returned no text blocks")
        return "".join(texts)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
