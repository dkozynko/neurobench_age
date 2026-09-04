#!/usr/bin/env bash
set -euo pipefail

# Predeclared finalist runner.  It waits for validation confirmation, checks
# the frozen gate, and opens the sealed test only for the single finalist.
PY=/venv/main/bin/python
SCRIPT=/workspace/neurobench_age_article_ready_20260904/official_reve_subset.py
ANALYZER=/workspace/neurobench_age_article_ready_20260904/scripts/analyze_paper_evidence.py
MANIFEST=/workspace/neurobench_age_layer_selection_20260902/manifest/age_medium_1000_nested_20260826.csv
DATA=/workspace/neurobench_data_hbn_1000_20260902
CONFIRMATION_SERVICE=neurobench_article_ready_confirmation_medium_1000_20260904
CONFIRMATION_ROOT=/workspace/results/article_ready_confirmation_medium_1000_20260904
GRID_SELECTION=/workspace/results/article_ready_augmentation_grid_1000_20260904/grid_selection.json
SCREEN_BASELINE=/workspace/results/article_ready_validation_cycle_1000_20260904/mean_linear/mean_linear
SCREEN_CANDIDATE=/workspace/results/article_ready_augmentation_grid_1000_20260904/medium_lambda/mean_linear
ROOT=/workspace/results/article_ready_sealed_final_medium_1000_20260904
CANDIDATE="$ROOT/finalist_medium_lambda"
GATE="$ROOT/final_gate.json"
BASELINE_FINAL=/workspace/results/layer_finalist_mean_linear_1000_20260903/mean_linear

mkdir -p "$ROOT"

echo "[article-ready-final] waiting for validation confirmation $(date -Is)"
while supervisorctl status "$CONFIRMATION_SERVICE" 2>/dev/null | grep -Eq "STARTING|RUNNING|STOPPING"; do
  sleep 30
done

mkdir -p "$ROOT/pre_test_audit"
echo "[article-ready-final] auditing all validation evidence before opening test"
"$PY" "$ANALYZER" \
  "$SCREEN_BASELINE/seed33" \
  "$CONFIRMATION_ROOT/matched_mean_linear/mean_linear/seed34" \
  "$CONFIRMATION_ROOT/matched_mean_linear/mean_linear/seed35" \
  --output-dir "$ROOT/pre_test_audit/baseline"
"$PY" "$ANALYZER" \
  "$SCREEN_CANDIDATE/seed33" \
  "$CONFIRMATION_ROOT/medium_lambda/mean_linear/seed34" \
  "$CONFIRMATION_ROOT/medium_lambda/mean_linear/seed35" \
  --output-dir "$ROOT/pre_test_audit/candidate"
"$PY" "$ANALYZER" \
  "$SCREEN_CANDIDATE/seed33" \
  "$CONFIRMATION_ROOT/medium_lambda/mean_linear/seed34" \
  "$CONFIRMATION_ROOT/medium_lambda/mean_linear/seed35" \
  --baseline-run "$SCREEN_BASELINE/seed33" \
  --baseline-run "$CONFIRMATION_ROOT/matched_mean_linear/mean_linear/seed34" \
  --baseline-run "$CONFIRMATION_ROOT/matched_mean_linear/mean_linear/seed35" \
  --output-dir "$ROOT/pre_test_audit/paired"

if "$PY" - "$GATE" "$GRID_SELECTION" <<'PY'
import json
import math
import sys
from pathlib import Path

gate_path = Path(sys.argv[1])
grid_selection_path = Path(sys.argv[2])
screen_baseline = Path("/workspace/results/article_ready_validation_cycle_1000_20260904/mean_linear/mean_linear")
screen_candidate = Path("/workspace/results/article_ready_augmentation_grid_1000_20260904/medium_lambda/mean_linear")
confirmation_root = Path("/workspace/results/article_ready_confirmation_medium_1000_20260904")

baseline_paths = {
    33: screen_baseline / "seed33",
    34: confirmation_root / "matched_mean_linear/mean_linear/seed34",
    35: confirmation_root / "matched_mean_linear/mean_linear/seed35",
}
candidate_paths = {
    33: screen_candidate / "seed33",
    34: confirmation_root / "medium_lambda/mean_linear/seed34",
    35: confirmation_root / "medium_lambda/mean_linear/seed35",
}

errors = []
rows = []
if not grid_selection_path.is_file():
    errors.append(f"missing frozen grid selection: {grid_selection_path}")
else:
    grid_selection = json.loads(grid_selection_path.read_text())
    winner = grid_selection.get("winner")
    if not isinstance(winner, dict) or winner.get("label") != "medium_lambda":
        errors.append("medium_lambda is not the frozen grid winner")
    if grid_selection.get("selection_metric") != "validation_pearsonr":
        errors.append("grid selection metric is not validation_pearsonr")
for seed in (33, 34, 35):
    seed_row = {"seed": seed}
    for label, paths in (("baseline", baseline_paths), ("candidate", candidate_paths)):
        run_dir = paths[seed]
        manifest_path = run_dir / "run_manifest.json"
        selection_path = run_dir / "selection.json"
        report_path = run_dir / "report.json"
        complexity_path = run_dir / "complexity.json"
        failure_paths = list(run_dir.glob("failure*.json"))
        test_paths = list(run_dir.glob("test_*.json"))
        test_prediction_path = run_dir / "predictions" / "test.jsonl"
        test_metric_path = run_dir / "epoch_test_metrics.jsonl"
        if (
            not manifest_path.is_file()
            or not selection_path.is_file()
            or not report_path.is_file()
            or not complexity_path.is_file()
        ):
            errors.append(f"{label} seed {seed}: missing required evidence")
            continue
        manifest = json.loads(manifest_path.read_text())
        selection = json.loads(selection_path.read_text())
        report = json.loads(report_path.read_text())
        complexity = json.loads(complexity_path.read_text())
        if manifest.get("status") != "completed":
            errors.append(f"{label} seed {seed}: manifest status is not completed")
        if manifest.get("evaluation_mode") != "validation_only":
            errors.append(f"{label} seed {seed}: validation evidence has the wrong evaluation mode")
        if manifest.get("test_access") != "sealed":
            errors.append(f"{label} seed {seed}: validation evidence does not seal the test")
        if manifest.get("missing"):
            errors.append(f"{label} seed {seed}: manifest has missing evidence")
        if report.get("evidence_status") != "complete":
            errors.append(f"{label} seed {seed}: report evidence is not complete")
        if report.get("test_status") != "withheld":
            errors.append(f"{label} seed {seed}: validation test is not withheld")
        if complexity.get("parameter_accounting_status") != "complete":
            errors.append(f"{label} seed {seed}: parameter accounting is incomplete")
        buckets = complexity.get("parameter_buckets")
        if not isinstance(buckets, dict) or not buckets:
            errors.append(f"{label} seed {seed}: parameter buckets are missing")
        if failure_paths or test_paths or test_prediction_path.is_file() or test_metric_path.is_file():
            if test_prediction_path.is_file():
                errors.append(f"{label} seed {seed}: validation evidence contains predictions/test.jsonl")
            errors.append(f"{label} seed {seed}: failure/test sentinel exists")
        value = selection.get("selected_val_pearsonr")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{label} seed {seed}: invalid selected validation Pearson")
            continue
        seed_row[f"{label}_selected_val_pearsonr"] = float(value)
    if "baseline_selected_val_pearsonr" in seed_row and "candidate_selected_val_pearsonr" in seed_row:
        seed_row["delta"] = seed_row["candidate_selected_val_pearsonr"] - seed_row["baseline_selected_val_pearsonr"]
        rows.append(seed_row)

if len(rows) != 3:
    errors.append("confirmation gate requires all three seeds")
deltas = [row["delta"] for row in rows]
wins = sum(delta > 0.0 for delta in deltas)
mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
worst_delta = min(deltas) if deltas else float("nan")
passed = not errors and mean_delta > 0.0 and wins >= 2 and worst_delta >= -0.02
payload = {
    "schema_version": 1,
    "status": "passed" if passed else "rejected",
    "gate": {
        "mean_delta_positive": True,
        "minimum_wins": 2,
        "minimum_worst_seed_delta": -0.02,
    },
    "mean_delta": mean_delta,
    "wins": wins,
    "losses": len(deltas) - wins,
    "worst_seed_delta": worst_delta,
    "per_seed": rows,
    "errors": errors,
    "finalist": "medium_lambda",
    "test_status_before_final": "withheld",
}
gate_path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
if not passed:
    raise SystemExit(2)
PY
then
  gate_rc=0
else
  gate_rc=$?
fi
if [ "$gate_rc" -ne 0 ]; then
  echo "[article-ready-final] validation gate rejected finalist rc=$gate_rc"
  exit "$gate_rc"
fi

echo "[article-ready-final] gate passed; starting the single frozen finalist test $(date -Is)"
if "$PY" "$SCRIPT" \
  --manifest "$MANIFEST" \
  --data-root "$DATA" \
  --output-dir "$CANDIDATE" \
  --config "$CANDIDATE/config.json" \
  --head-variant mean_linear \
  --evaluation-protocol strict \
  --deterministic \
  --evaluation-mode final_test \
  --allow-sealed-test-evaluation \
  --augmentation-consistency \
  --augmentation-consistency-lambda 0.02 \
  --augmentation-noise-scale 0.005 \
  --seeds 33 34 35; then
  run_rc=0
else
  run_rc=$?
fi
if [ "$run_rc" -ne 0 ]; then
  echo "[article-ready-final] finalist run failed rc=$run_rc"
  exit "$run_rc"
fi

if "$PY" "$ANALYZER" \
  "$CANDIDATE/mean_linear/seed33" \
  "$CANDIDATE/mean_linear/seed34" \
  "$CANDIDATE/mean_linear/seed35" \
  --output-dir "$ROOT/analysis"; then
  analysis_rc=0
else
  analysis_rc=$?
fi
if [ "$analysis_rc" -ne 0 ]; then
  echo "[article-ready-final] finalist evidence analysis failed rc=$analysis_rc"
  exit "$analysis_rc"
fi

if [ -f "$BASELINE_FINAL/seed33/report.json" ]; then
  "$PY" - "$ROOT/final_test_comparison.json" <<'PY'
import json
import math
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
baseline_root = Path("/workspace/results/layer_finalist_mean_linear_1000_20260903/mean_linear")
candidate_root = Path("/workspace/results/article_ready_sealed_final_medium_1000_20260904/finalist_medium_lambda/mean_linear")
rows = []
errors = []
for seed in (33, 34, 35):
    baseline = json.loads((baseline_root / f"seed{seed}/report.json").read_text())
    candidate = json.loads((candidate_root / f"seed{seed}/report.json").read_text())
    baseline_test = baseline.get("test_pearsonr")
    candidate_test = candidate.get("test_pearsonr")
    baseline_val = baseline.get("selected_val_pearsonr")
    candidate_val = candidate.get("selected_val_pearsonr")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in (baseline_test, candidate_test, baseline_val, candidate_val)):
        errors.append(f"invalid metrics for seed {seed}")
        continue
    rows.append({
        "seed": seed,
        "baseline_selected_val_pearsonr": float(baseline_val),
        "candidate_selected_val_pearsonr": float(candidate_val),
        "validation_delta": float(candidate_val - baseline_val),
        "baseline_test_pearsonr": float(baseline_test),
        "candidate_test_pearsonr": float(candidate_test),
        "test_delta": float(candidate_test - baseline_test),
    })
test_deltas = [row["test_delta"] for row in rows]
val_deltas = [row["validation_delta"] for row in rows]
test_mean = sum(test_deltas) / len(test_deltas) if test_deltas else float("nan")
val_mean = sum(val_deltas) / len(val_deltas) if val_deltas else float("nan")
test_sd = math.sqrt(sum((value - test_mean) ** 2 for value in test_deltas) / (len(test_deltas) - 1)) if len(test_deltas) > 1 else float("nan")
payload = {
    "schema_version": 1,
    "status": "complete" if len(rows) == 3 and not errors else "invalid",
    "candidate": "medium_lambda",
    "per_seed": rows,
    "validation_mean_delta": val_mean,
    "test_mean_delta": test_mean,
    "test_sample_sd_delta": test_sd,
    "test_wins": sum(value > 0 for value in test_deltas),
    "test_losses": sum(value < 0 for value in test_deltas),
    "test_ties": sum(value == 0 for value in test_deltas),
    "errors": errors,
    "paired_bootstrap": {
        "status": "unavailable",
        "reason": "historical matched baseline has no subject-level test prediction file",
    },
    "baseline_evidence_scope": "historical sealed test report and markers",
    "candidate_evidence_scope": "article-ready sealed test report, markers, and subject-level predictions",
}
output_path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
if payload["status"] != "complete":
    raise SystemExit(2)
PY
else
  echo "[article-ready-final] final comparison skipped: matched baseline report not found"
  exit 2
fi
