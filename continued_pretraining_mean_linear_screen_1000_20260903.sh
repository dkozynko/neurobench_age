#!/bin/bash
set -euo pipefail

PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_layer_selection_20260902/official_reve_subset.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
BASE=/workspace/results/continued_pretraining_mean_linear_screen_1000_20260903

"$PY" "$SCRIPT" \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$BASE" \
  --config "$BASE/config.json" \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --continued-pretraining \
  --pretraining-epochs 1 \
  --pretraining-mask-fraction 0.15 \
  --pretraining-mask-block-samples 20 \
  --pretraining-learning-rate 1e-5 \
  --pretraining-weight-decay 0.05 \
  --seeds 33
