#!/usr/bin/env bash
set -euo pipefail
unset VLLM_PORT
exec "${PYTHON:-python3}" -m vllm.entrypoints.openai.api_server \
  --model "${EMBEDDING_MODEL_PATH:-Qwen/Qwen3-Embedding-8B}" \
  --served-model-name Qwen3-Embedding-8B --task embed \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8001}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" "$@"
