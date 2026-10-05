#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  download|filter|embed) module="$1"; shift ;;
  *) echo 'Usage: bash scripts/build_corpus.sh {download|filter|embed} [options]' >&2; exit 2 ;;
esac
exec "${PYTHON:-python3}" -m "corpus.$module" "$@"
