#!/bin/bash
set -euo pipefail

PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_layer_selection_20260902/official_reve_subset.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
BASE=/workspace/results/augmentation_consistency_mean_linear_screen_1000_20260903

"$PY" "$SCRIPT" \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$BASE" \
  --config "$BASE/config.json" \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --augmentation-consistency \
  --augmentation-consistency-lambda 0.05 \
  --augmentation-noise-scale 0.01 \
  --seeds 33
