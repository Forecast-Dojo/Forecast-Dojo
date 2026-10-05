#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-memory-free}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    memory-free|memory-on|no-tools) ;;
    -h|--help)
        echo "Usage: bash scripts/evaluate.sh {memory-free|memory-on|no-tools} [evaluation options]"
        echo "Example: bash scripts/evaluate.sh memory-free --config configs/eval/qwen3.5.json --index-root /datasets/news-index"
        exit 0
        ;;
    *) echo "Unknown forecast mode: $MODE" >&2; exit 2 ;;
esac

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m evaluation.run --mode "$MODE" "$@"
