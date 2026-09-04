#!/usr/bin/env bash
set -euo pipefail

# Portable validation-only launcher for the article pipeline.  Final-test
# evaluation is deliberately a separate, manually reviewed operation.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:?Set MANIFEST to the canonical article manifest}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the prepared HBN root}"
PHASE="${PHASE:-baseline}"
HEAD_VARIANT="${HEAD_VARIANT:-mean_linear}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/canonical/runs/$PHASE}"
SEEDS_TEXT="${SEEDS:-33 34 35}"

case "$PHASE" in
  baseline|screen)
    DEFAULT_CONFIG="$REPO_ROOT/configs/article/validation_cycle.json"
    ;;
  confirmation)
    DEFAULT_CONFIG="$REPO_ROOT/configs/article/confirmation.json"
    ;;
  *)
    echo "unsupported article validation phase: $PHASE" >&2
    exit 2
    ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

for arg in "$@"; do
  case "$arg" in
    --evaluation-mode|--evaluation-mode=*|--strict-final-test*|--allow-sealed-test-evaluation*)
      echo "validation launcher rejects final-test override: $arg" >&2
      exit 2
      ;;
  esac
done

read -r -a SEED_LIST <<< "$SEEDS_TEXT"
if [ "${#SEED_LIST[@]}" -eq 0 ]; then
  echo "SEEDS must contain at least one integer" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m neurobench_age.pipelines.official \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --config "$CONFIG" \
  --head-variant "$HEAD_VARIANT" \
  --evaluation-protocol strict \
  --deterministic \
  --evaluation-mode validation_only \
  --seeds "${SEED_LIST[@]}" \
  "$@"
