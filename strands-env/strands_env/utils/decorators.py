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

"""Decorator and utility function helpers for `strands_env`."""

from __future__ import annotations

import asyncio
import ctypes
import inspect
import os
import threading
from collections.abc import Callable
from functools import wraps
from typing import Any


def requires_env(*env_vars: str) -> Callable[..., Any]:
    """Decorator that validates environment variables at call time."""

    def _check() -> str | None:
        missing = [v for v in env_vars if not os.getenv(v)]
        return f"Error: missing required environment variable(s): {', '.join(missing)}" if missing else None

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if err := _check():
                    return err
                return await fn(*args, **kwargs)

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if err := _check():
                raise OSError(err)
            return fn(*args, **kwargs)

        return sync_wrapper

    return decorator


def cache_by(*key_args: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache results using only the named arguments as the key."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        cache: dict[tuple, Any] = {}
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            key = tuple(bound.arguments[k] for k in key_args)
            if key not in cache:
                cache[key] = fn(*args, **kwargs)
            return cache[key]

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


class TimeoutInterrupt(BaseException):
    """Injected into a timed-out thread to interrupt it."""


def with_timeout(timeout: float | None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that enforces a timeout on function execution."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if timeout is None:
            return func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result: list[Any] = []
            exception: list[BaseException] = []

            def target() -> None:
                try:
                    result.append(func(*args, **kwargs))
                except BaseException as e:
                    exception.append(e)

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                if thread.ident is not None:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(thread.ident), ctypes.py_object(TimeoutInterrupt)
                    )
                raise TimeoutError(f"Operation timed out after {timeout} seconds")

            if exception:
                raise exception[0]
            return result[0]

        return wrapper

    return decorator
