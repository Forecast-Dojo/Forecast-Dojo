#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python3}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH:?Set MODEL_PATH to the model ID or checkpoint path}" \
  --served-model-name "${MODEL_NAME:-$MODEL_PATH}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-30000}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:?Set the model-specific vLLM tool-call parser}" "$@"
