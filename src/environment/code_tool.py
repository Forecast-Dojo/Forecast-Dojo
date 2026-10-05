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

"""Local `python` tool backed by a subprocess Python interpreter."""

import asyncio
import logging
import os
from pathlib import Path

from strands import tool

from .subprocess_interpreter import SubprocessInterpreter

logger = logging.getLogger(__name__)


class LocalPythonToolkit:
    """`python` tool that runs Python code in a local subprocess."""

    DEFAULT_EXECUTION_TIMEOUT_SECONDS = 10

    _sandbox_verified: bool = False

    def __init__(
        self,
        execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        concurrency: asyncio.Semaphore | int = 24,
        memory_limit_mb: int = 4096,
    ) -> None:
        network_isolation = os.environ.get("FORECAST_PYTHON_SANDBOX", "on").strip().lower() not in ("0", "off", "false", "no")
        if network_isolation:
            if not LocalPythonToolkit._sandbox_verified:
                repo_root = Path(__file__).resolve().parents[2]
                hidden = [str(repo_root / d) for d in ("data", "src", "prompts")]
                hidden += [p for p in (os.environ.get("INDEX_ROOT"), os.environ.get("FORECAST_INDEX_ROOT")) if p]
                probe = SubprocessInterpreter.assert_sandboxed(hidden_paths=hidden)
                logger.info("python tool sandbox self-test passed (%s)", probe.replace("\n", " "))
                LocalPythonToolkit._sandbox_verified = True
        else:
            logger.error("FORECAST_PYTHON_SANDBOX=off: python tool runs WITH network access; results are not valid benchmark numbers")

        self._interpreter = SubprocessInterpreter(
            require_confirm=False,
            print_stdout=False,
            print_stderr=False,
            execution_timeout=execution_timeout,
            memory_limit_mb=memory_limit_mb,
            network_isolation=network_isolation,
        )
        self._semaphore = concurrency if isinstance(concurrency, asyncio.Semaphore) else asyncio.Semaphore(concurrency)

    @tool
    async def python(self, code: str) -> str:
        """Execute Python code and return its printed output.

        numpy, pandas and scipy are available. The process is killed after
        10 seconds, so keep each script short.

        The code runs in a fresh Python process each call: variables do NOT
        persist between calls, so send one self-contained script per call and
        `print` every value you need back. A bare final expression is echoed
        automatically, REPL-style. Type in the numbers from your research —
        nothing from a previous call is in scope.

        Args:
            code: The Python code to execute.

        Returns:
            Captured stdout, plus stderr and the exit code when execution
            fails or times out.
        """

        if not isinstance(code, str):
            code = str(code)
        async with self._semaphore:
            try:
                return await asyncio.to_thread(self._interpreter.run, code, "python")
            except Exception as e:
                logger.warning("python tool failed: %s", e)
                return f"python tool failed: {e}"
