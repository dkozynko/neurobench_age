#!/bin/bash
set -euo pipefail

exec /venv/main/bin/python \
  /workspace/neurobench_age_1000_20260902/download_hbn_resting_direct.py \
  --data-root /workspace/neurobench_data_hbn_1000_20260902 \
  --manifest /workspace/hbn_subset/manifests/age_medium_1000_nested_20260826.csv \
  --releases R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 \
  --workers 8
