#!/bin/bash
set -euo pipefail

PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_layer_selection_20260902/official_reve_subset.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
BASE=/workspace/results/mean_layer_mix_fixed_screen_1000_20260903

for alpha_spec in "025 0.25" "050 0.5" "100 1.0"; do
  set -- $alpha_spec
  LABEL=$1
  ALPHA=$2
  OUT="$BASE/alpha_$LABEL"
  "$PY" "$SCRIPT" \
    --manifest "$MANIFEST" \
    --data-root "$DATA" \
    --output-dir "$OUT" \
    --config "$OUT/config.json" \
    --head-variant mean_layer_mix_fixed \
    --layer-indices -2 -1 \
    --layer-mix-alpha "$ALPHA" \
    --evaluation-protocol strict \
    --seeds 33
done
