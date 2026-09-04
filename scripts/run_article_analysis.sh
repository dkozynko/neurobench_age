#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ANALYZER="${ANALYZER:-$REPO_ROOT/scripts/analyze_paper_evidence.py}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 RUN_DIR [RUN_DIR ...] [--output-dir DIR]" >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ANALYZER" "$@"
