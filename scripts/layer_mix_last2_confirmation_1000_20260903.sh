#!/bin/bash
set -euo pipefail

PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_layer_selection_20260902/official_reve_subset.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
BASE=/workspace/results/layer_mix_last2_confirmation_1000_20260903
CONFIG=$BASE/config.json

exec "$PY" "$SCRIPT" \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$BASE" \
  --config "$CONFIG" \
  --head-variant mean_layer_mix \
  --layer-indices -2 -1 \
  --evaluation-protocol strict \
  --seeds 34 35
