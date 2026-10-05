"""Configure official provider APIs and local model servers."""

from __future__ import annotations

import argparse
import itertools
import os
from typing import Any

from strands_env.core.models import (
    ModelFactory,
    anthropic_model_factory,
    check_server_health,
    get_model_id,
    openai_compatible_model_factory,
    openai_model_factory,
    sglang_model_factory,
)


def build_model_factory(
    args: argparse.Namespace, sampling_params: dict[str, Any],
) -> tuple[ModelFactory, str]:
    params = dict(sampling_params)
    if args.backend == "sglang":
        from strands_sglang import get_client, get_tokenizer
        from strands_sglang.tool_parsers import get_tool_parser

        check_server_health(args.base_url)
        model_id = args.model_id or get_model_id(args.base_url)
        parser = args.tool_parser or ("glm" if "glm" in model_id.lower() else "hermes")
        factory = sglang_model_factory(
            client=get_client(
                args.base_url, max_connections=args.max_connections,
                timeout=args.request_timeout,
            ),
            tokenizer=get_tokenizer(args.tokenizer_path or model_id),
            tool_parser=get_tool_parser(parser),
            sampling_params=params,
        )
        return factory, model_id

    key_env = args.api_key_env or {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
    }[args.backend]
    api_key = os.environ.get(key_env)
    if not api_key and args.backend in ("anthropic", "openai"):
        raise SystemExit(f"Set {key_env} to use the {args.backend} API")
    client_args = {
        "api_key": api_key or "EMPTY",
        "timeout": args.request_timeout,
        "max_retries": 5,
    }

    if args.backend == "anthropic":
        if args.no_thinking:
            params["thinking"] = {"type": "disabled"}
        elif args.reasoning_effort:
            params["thinking"] = {"type": "adaptive"}
        if args.reasoning_effort:
            params["output_config"] = {
                **params.get("output_config", {}), "effort": args.reasoning_effort,
            }
        factory_fn = anthropic_model_factory
    elif args.backend in ("openai", "openai-compatible"):
        if args.reasoning_effort:
            params["reasoning_effort"] = args.reasoning_effort
        factory_fn = (
            openai_model_factory if args.backend == "openai"
            else openai_compatible_model_factory
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    urls = [url.strip() for url in args.base_url.split(",") if url.strip()]
    factories = [
        factory_fn(
            model_id=args.model_id,
            sampling_params=params,
            client_args={**client_args, "base_url": url},
        )
        for url in urls
    ]
    if len(factories) == 1:
        return factories[0], args.model_id
    replicas = itertools.cycle(factories)

    def factory() -> Any:
        return next(replicas)()

    return factory, args.model_id
