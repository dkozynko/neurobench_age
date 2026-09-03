#!/bin/bash
set -euo pipefail

PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_layer_selection_20260902/official_reve_subset.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
BASE=/workspace/results/layer_finalist_mean_linear_1000_20260903
CONFIG=$BASE/config.json

exec "$PY" "$SCRIPT" \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$BASE" \
  --config "$CONFIG" \
  --head-variant mean_linear \
  --layer-index -1 \
  --evaluation-protocol strict \
  --strict-final-test \
  --seeds 33 34 35
