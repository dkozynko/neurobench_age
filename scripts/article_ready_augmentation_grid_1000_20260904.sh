#!/usr/bin/env bash
set -euo pipefail

# This queued screen starts only after the primary article-ready cycle exits.
# It is validation-only and deliberately has no sealed-test flag.
PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_article_ready_20260904/official_reve_subset.py
ANALYZER=/workspace/neurobench_age_article_ready_20260904/scripts/analyze_paper_evidence.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
PRIMARY_SERVICE=neurobench_article_ready_validation_cycle_1000_20260904
BASELINE=/workspace/results/article_ready_validation_cycle_1000_20260904/mean_linear/mean_linear/seed33
ROOT=/workspace/results/article_ready_augmentation_grid_1000_20260904
POLICY="$ROOT/grid_policy.json"

mkdir -p "$ROOT"

# This is a screening-only grid.  The policy is written beside the results so
# the winner-selection rule is part of the evidence package, not an implicit
# property of this shell script.
"$PY" - "$POLICY" <<'PY'
import json
import sys
from pathlib import Path

policy_path = Path(sys.argv[1])
payload = {
    "schema_version": "1.0",
    "screening_only": True,
    "selection_metric": "validation_pearsonr",
    "selection_rule": "select the single candidate with maximum seed-33 validation Pearson",
    "candidates": [
        {"label": "mild_lambda", "lambda": 0.01, "noise_scale": 0.005},
        {"label": "medium_lambda", "lambda": 0.02, "noise_scale": 0.005},
        {"label": "strong_lambda", "lambda": 0.05, "noise_scale": 0.005},
    ],
    "confirmation_rule": "only the frozen winner may proceed to confirmation seeds",
    "sealed_test_rule": "no candidate from this screen may access the sealed test split",
}
policy_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[article-ready-grid] waiting for primary cycle $(date -Is)"
while supervisorctl status "$PRIMARY_SERVICE" 2>/dev/null | grep -Eq "STARTING|RUNNING|STOPPING"; do
  sleep 30
done

status=0
run_case() {
  local label=$1
  local lambda=$2
  local noise=$3
  local output="$ROOT/$label"
  echo "[article-ready-grid] START $label lambda=$lambda noise=$noise $(date -Is)"
  if "$PY" "$SCRIPT" \
    --manifest "$MANIFEST" \
    --data-root "$DATA" \
    --output-dir "$output" \
    --config "$output/config.json" \
    --head-variant mean_linear \
    --evaluation-protocol strict \
    --deterministic \
    --evaluation-mode validation_only \
    --augmentation-consistency \
    --augmentation-consistency-lambda "$lambda" \
    --augmentation-noise-scale "$noise" \
    --seeds 33; then
    local rc=0
  else
    local rc=$?
  fi
  echo "[article-ready-grid] END $label rc=$rc $(date -Is)"
  [ "$rc" -eq 0 ] || status=1
}

run_case mild_lambda 0.01 0.005
run_case medium_lambda 0.02 0.005
run_case strong_lambda 0.05 0.005

for label in mild_lambda medium_lambda strong_lambda; do
  candidate="$ROOT/$label/mean_linear/seed33"
  analysis="$ROOT/$label/analysis"
  if [ -f "$BASELINE/run_manifest.json" ] && [ -f "$candidate/run_manifest.json" ]; then
    "$PY" "$ANALYZER" "$BASELINE" "$candidate" \
      --output-dir "$analysis" \
      --baseline-run "$BASELINE" || status=1
  else
    echo "[article-ready-grid] analysis skipped for $label: incomplete evidence"
    status=1
  fi
done

# Rank candidates from their independently audited validation evidence.  This
# writes a complete, reproducible decision record but launches no confirmation
# or sealed-test evaluation.
"$PY" - "$ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for label in ("mild_lambda", "medium_lambda", "strong_lambda"):
    evidence_path = root / label / "analysis" / "candidate_vs_baseline.json"
    if not evidence_path.is_file():
        raise SystemExit(f"missing candidate evidence: {evidence_path}")
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != 1:
        raise SystemExit(f"candidate {label} does not have exactly one matched seed")
    row = per_seed[0]
    score = row.get("candidate_pearsonr")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        raise SystemExit(f"candidate {label} has no finite validation Pearson")
    rows.append(
        {
            "label": label,
            "seed": row["seed"],
            "validation_pearsonr": float(score),
            "baseline_validation_pearsonr": row.get("baseline_pearsonr"),
            "delta": row.get("pearsonr_delta"),
        }
    )

winner = max(rows, key=lambda row: row["validation_pearsonr"])
selection = {
    "schema_version": "1.0",
    "selection_metric": "validation_pearsonr",
    "selection_rule": "argmax over the three predeclared seed-33 candidates",
    "candidates": rows,
    "winner": winner,
    "next_step": "freeze winner before confirmation seeds",
    "sealed_test_access": "withheld",
}
(root / "grid_selection.json").write_text(
    json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

exit "$status"
