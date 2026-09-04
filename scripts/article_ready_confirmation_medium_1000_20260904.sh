#!/usr/bin/env bash
set -euo pipefail

# Article-ready confirmation for the predeclared medium augmentation setting.
# This remains validation-only; the sealed test is owned by a later frozen
# finalist invocation and is intentionally unreachable here.
PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_article_ready_20260904/official_reve_subset.py
ANALYZER=/workspace/neurobench_age_article_ready_20260904/scripts/analyze_paper_evidence.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
ROOT=/workspace/results/article_ready_confirmation_medium_1000_20260904
BASELINE="$ROOT/matched_mean_linear"
CANDIDATE="$ROOT/medium_lambda"
ANALYSIS="$ROOT/analysis"
PRIMARY_BASELINE=/workspace/results/article_ready_validation_cycle_1000_20260904/mean_linear/mean_linear
SCREEN_CANDIDATE=/workspace/results/article_ready_augmentation_grid_1000_20260904/medium_lambda/mean_linear
GRID_SELECTION=/workspace/results/article_ready_augmentation_grid_1000_20260904/grid_selection.json

mkdir -p "$ROOT"

# Do not spend confirmation compute unless the predeclared seed-33 screen
# actually selected this exact candidate.
"$PY" - "$GRID_SELECTION" <<'PY'
import json
import sys
from pathlib import Path

selection_path = Path(sys.argv[1])
if not selection_path.is_file():
    raise SystemExit(f"missing frozen grid selection: {selection_path}")
selection = json.loads(selection_path.read_text(encoding="utf-8"))
winner = selection.get("winner")
if not isinstance(winner, dict) or winner.get("label") != "medium_lambda":
    raise SystemExit("medium_lambda is not the frozen grid winner")
if selection.get("selection_metric") != "validation_pearsonr":
    raise SystemExit("grid selection metric is not validation_pearsonr")
PY

run_validation() {
  local name=$1
  shift
  echo "[article-ready-confirmation] START ${name} $(date -Is)"
  "$PY" "$SCRIPT" "$@"
  local rc=$?
  echo "[article-ready-confirmation] END ${name} rc=${rc} $(date -Is)"
  return "$rc"
}

status=0

# Re-run the matched controls for seeds 34/35 under the corrected evidence
# writer.  This makes complexity metadata complete for the confirmation set
# instead of inheriting the earlier missing UID-folder warning.
run_validation matched_baseline_34_35 \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$BASELINE" \
  --config "$BASELINE/config.json" \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --deterministic \
  --evaluation-mode validation_only \
  --seeds 34 35 || status=1

# Confirm the predeclared medium setting on the two held-back confirmation
# seeds.  The noisy paired view remains train-only and uses the exact screen
# setting that won the seed-33 validation screen.
run_validation medium_lambda_34_35 \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$CANDIDATE" \
  --config "$CANDIDATE/config.json" \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --deterministic \
  --evaluation-mode validation_only \
  --augmentation-consistency \
  --augmentation-consistency-lambda 0.02 \
  --augmentation-noise-scale 0.005 \
  --seeds 34 35 || status=1

for seed in 33 34 35; do
  baseline="$PRIMARY_BASELINE/seed${seed}"
  candidate="$SCREEN_CANDIDATE/seed33"
  if [ "$seed" -ne 33 ]; then
    baseline="$BASELINE/mean_linear/seed${seed}"
    candidate="$CANDIDATE/mean_linear/seed${seed}"
  fi
  if [ -f "$baseline/run_manifest.json" ] && [ -f "$candidate/run_manifest.json" ]; then
    :
  else
    echo "[article-ready-confirmation] missing evidence for seed ${seed}"
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  # Keep baseline and candidate groups disjoint.  Each seed appears once in
  # each group, so the analyzer can perform exact matched comparisons.
  if ! "$PY" "$ANALYZER" \
    "$PRIMARY_BASELINE/seed33" \
    "$BASELINE/mean_linear/seed34" \
    "$BASELINE/mean_linear/seed35" \
    --output-dir "$ANALYSIS/baseline"; then
    status=1
  fi
  if ! "$PY" "$ANALYZER" \
    "$SCREEN_CANDIDATE/seed33" \
    "$CANDIDATE/mean_linear/seed34" \
    "$CANDIDATE/mean_linear/seed35" \
    --output-dir "$ANALYSIS/candidate_vs_baseline" \
    --baseline-run "$PRIMARY_BASELINE/seed33" \
    --baseline-run "$BASELINE/mean_linear/seed34" \
    --baseline-run "$BASELINE/mean_linear/seed35"; then
    status=1
  fi
else
  echo "[article-ready-confirmation] analysis skipped: incomplete evidence"
fi

exit "$status"
