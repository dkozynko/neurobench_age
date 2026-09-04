#!/usr/bin/env bash
set -euo pipefail

# Article-ready validation-only cycle.  The final test is intentionally not
# reachable from this script; only a separately frozen finalist may run it.
PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_article_ready_20260904/official_reve_subset.py
ANALYZER=/workspace/neurobench_age_article_ready_20260904/scripts/analyze_paper_evidence.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
ROOT=/workspace/results/article_ready_validation_cycle_1000_20260904
BASELINE="$ROOT/mean_linear"
CONTINUED="$ROOT/continued_pretraining_mean_linear"
AUGMENTED="$ROOT/augmentation_consistency_mean_linear"
ANALYSIS="$ROOT/analysis"

mkdir -p "$ROOT"

run_validation() {
  local name=$1
  shift
  echo "[article-ready] START ${name} $(date -Is)"
  "$PY" "$SCRIPT" "$@"
  local rc=$?
  echo "[article-ready] END ${name} rc=${rc} $(date -Is)"
  return "$rc"
}

status=0

# Three matched baseline seeds provide the reference distribution.  No test
# loader is opened because --evaluation-mode defaults to validation_only.
run_validation baseline \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$BASELINE" \
  --config "$BASELINE/config.json" \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --deterministic \
  --evaluation-mode validation_only \
  --seeds 33 34 35 || status=1

# Continued pretraining is screened on seed 33 first.  The train-only
# objective never sees age labels and never opens the held-out test split.
run_validation continued_pretraining \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$CONTINUED" \
  --config "$CONTINUED/config.json" \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --deterministic \
  --evaluation-mode validation_only \
  --continued-pretraining \
  --pretraining-epochs 1 \
  --pretraining-mask-fraction 0.15 \
  --pretraining-mask-block-samples 20 \
  --pretraining-learning-rate 1e-5 \
  --pretraining-weight-decay 0.05 \
  --seeds 33 || status=1

# Augmentation consistency is also a seed-33 validation screen.  Keep the
# original/noisy pairing train-only and record it in the run manifest.
run_validation augmentation_consistency \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$AUGMENTED" \
  --config "$AUGMENTED/config.json" \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --deterministic \
  --evaluation-mode validation_only \
  --augmentation-consistency \
  --augmentation-consistency-lambda 0.05 \
  --augmentation-noise-scale 0.01 \
  --seeds 33 || status=1

analyze_if_complete() {
  local label=$1
  shift
  local missing=0
  local run_dir
  for run_dir in "$@"; do
    if [ ! -f "$run_dir/run_manifest.json" ]; then
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    echo "[article-ready] analysis skipped for ${label}: incomplete evidence"
    status=1
    return 0
  fi
  local output="$ANALYSIS/$label"
  local -a candidate_dirs=("$@")
  echo "[article-ready] ANALYZE ${label} $(date -Is)"
  if [ "$label" = "baseline" ]; then
    if ! "$PY" "$ANALYZER" "${candidate_dirs[@]}" --output-dir "$output"; then
      status=1
    fi
  else
    if ! "$PY" "$ANALYZER" "${candidate_dirs[@]}" \
      --output-dir "$output" \
      --baseline-run "$BASELINE/mean_linear/seed33"; then
      status=1
    fi
  fi
}

# Keep each scientific comparison class in its own analysis package.  In
# particular, do not put repeated seed 33 runs into one candidate group.
analyze_if_complete baseline \
  "$BASELINE/mean_linear/seed33" \
  "$BASELINE/mean_linear/seed34" \
  "$BASELINE/mean_linear/seed35"
analyze_if_complete continued_pretraining \
  "$CONTINUED/mean_linear/seed33"
analyze_if_complete augmentation_consistency \
  "$AUGMENTED/mean_linear/seed33"

exit "$status"
