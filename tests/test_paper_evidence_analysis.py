from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.analysis.paper_evidence import (
    audit_run,
    analyze_runs,
    complexity_adjusted_comparison,
    compute_age_group_metrics,
    paired_bootstrap_delta,
    summarize_seed_stability,
)
from neurobench_age.core.predictions import compute_regression_metrics


def _write_run(
    root: Path,
    *,
    variant: str,
    seed: int,
    hardware_class: str = "A100/40GB",
    predictions: list[tuple[str, float, float]] | None = None,
) -> Path:
    run_dir = root / variant / f"seed{seed}"
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0",
        "run_id": f"{variant}-seed{seed}",
        "status": "completed",
        "task": "age",
        "dataset_manifest": "dataset-sha",
        "split_fingerprint": "split-sha",
        "seed": seed,
        "config_hash": "config-sha",
        "source_tree_sha256": "source-sha",
        "protocol_digest": "protocol-sha",
        "deterministic_policy": "best_effort",
        "deterministic_policy_satisfied": True,
        "evaluation_mode": "validation_only",
        "test_access": "sealed",
        "comparison_config_hash": "comparison-sha",
        "comparison_factor_keys": ["head_variant"],
        "hardware": {"hardware_class": hardware_class},
        "missing": [],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest) + "\n")
    (run_dir / "complexity.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "parameter_accounting_status": "complete",
                "parameter_buckets": {
                    "encoder": {"total": 100, "trainable": 100, "frozen": 0},
                    "head": {"total": 10 if variant == "mean_linear" else 20, "trainable": 10 if variant == "mean_linear" else 20, "frozen": 0},
                    "auxiliary": [],
                    "total": {"total": 110 if variant == "mean_linear" else 120, "trainable": 110 if variant == "mean_linear" else 120, "frozen": 0},
                },
                "phases": {"train": {"elapsed_seconds": 10.0, "samples_per_second": 4.0}},
                "hardware": {"hardware_class": hardware_class},
            }
        )
        + "\n"
    )
    (run_dir / "config.json").write_text(json.dumps({"schema_version": "1.0", "config": {"head_variant": variant}}) + "\n")
    (run_dir / "report.json").write_text(json.dumps({"schema_version": "1.0", "status": "completed", "head_variant": variant}) + "\n")
    rows = predictions or [("s-1", 10.0, 11.0), ("s-2", 20.0, 19.0), ("s-3", 30.0, 31.0), ("s-4", 40.0, 38.0)]
    selected_score = compute_regression_metrics(
        [truth for _, truth, _ in rows], [pred for _, _, pred in rows]
    )["pearsonr"]
    (run_dir / "selection.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "selected_epoch": 2,
                "selected_val_pearsonr": selected_score,
            }
        )
        + "\n"
    )
    (run_dir / "epoch_validation_metrics.jsonl").write_text(
        "{\"schema_version\":\"1.0\",\"epoch\":1,\"val/pearsonr\":0.6}\n"
        + json.dumps(
            {
                "schema_version": "1.0",
                "epoch": 2,
                "val/pearsonr": selected_score,
            }
        )
        + "\n"
    )
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "validation.jsonl").write_text(
        "".join(
            json.dumps({"schema_version": "1.0", "subject_id": subject, "true_age": truth, "predicted_age": pred, "split": "validation", "seed": seed}) + "\n"
            for subject, truth, pred in rows
        )
    )
    return run_dir


def test_audit_rejects_invalid_parameter_artifact(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    payload = json.loads((run_dir / "complexity.json").read_text())
    payload["parameter_buckets"]["total"]["total"] = 111
    (run_dir / "complexity.json").write_text(json.dumps(payload) + "\n")

    audit = audit_run(run_dir)

    assert audit["status"] == "invalid"
    assert any("parameter" in error for error in audit["errors"])


def test_audit_rejects_head_complexity_bucket_mismatch(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    complexity_path = run_dir / "complexity.json"
    complexity = json.loads(complexity_path.read_text())
    complexity["head_complexity"] = {"parameter_count": 513}
    complexity_path.write_text(json.dumps(complexity) + "\n")

    audit = audit_run(run_dir)

    assert audit["status"] == "invalid"
    assert any("head complexity" in error for error in audit["errors"])


def test_audit_rejects_run_manifest_that_is_not_completed(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    audit = audit_run(run_dir)

    assert audit["status"] == "invalid"
    assert any("status" in error for error in audit["errors"])


def test_stability_reports_sd_wins_and_worst_seed_delta(tmp_path: Path) -> None:
    baseline = [_write_run(tmp_path, variant="mean_linear", seed=33), _write_run(tmp_path, variant="mean_linear", seed=34)]
    candidate = [
        _write_run(tmp_path, variant="candidate", seed=33, predictions=[("s-1", 10.0, 10.1), ("s-2", 20.0, 20.1), ("s-3", 30.0, 30.1), ("s-4", 40.0, 39.9)]),
        _write_run(tmp_path, variant="candidate", seed=34, predictions=[("s-1", 10.0, 14.0), ("s-2", 20.0, 16.0), ("s-3", 30.0, 34.0), ("s-4", 40.0, 36.0)]),
    ]

    summary = summarize_seed_stability(candidate, baseline)

    assert summary["seed_count"] == 2
    assert summary["pearsonr_sample_sd"] >= 0
    assert summary["pearsonr_wins"] == 1
    assert summary["worst_seed_delta"] < 0


def test_paired_bootstrap_and_train_only_age_groups_are_deterministic() -> None:
    baseline = {"s-1": (10.0, 10.0), "s-2": (20.0, 21.0), "s-3": (30.0, 29.0), "s-4": (40.0, 42.0)}
    candidate = {"s-1": (10.0, 10.0), "s-2": (20.0, 20.0), "s-3": (30.0, 30.0), "s-4": (40.0, 40.0)}

    bootstrap = paired_bootstrap_delta(candidate, baseline, metric="mae", iterations=200, seed=20260903)
    groups = compute_age_group_metrics(candidate, train_age_reference={"s-1": 10.0, "s-2": 20.0, "s-3": 30.0, "s-4": 40.0})

    assert bootstrap["seed"] == 20260903
    assert bootstrap["iterations"] == 200
    assert bootstrap["delta"] < 0
    assert bootstrap["ci_low"] <= bootstrap["ci_high"]
    assert {row["age_group"] for row in groups} == {"q1", "q2", "q3", "q4"}


def test_audit_marks_missing_predictions_as_partial(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    (run_dir / "predictions" / "validation.jsonl").unlink()

    audit = audit_run(run_dir)

    assert audit["status"] == "partial"
    assert any("prediction" in reason for reason in audit["missing"])


def test_audit_reconciles_final_test_marker_with_test_predictions(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    manifest["evaluation_mode"] = "final_test"
    manifest["test_access"] = "single_use_predeclared"
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest) + "\n")
    validation_rows = [
        json.loads(line)
        for line in (run_dir / "predictions" / "validation.jsonl").read_text().splitlines()
    ]
    test_rows = [dict(row, split="test") for row in validation_rows]
    (run_dir / "predictions" / "test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows)
    )
    (run_dir / "test_started.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evaluation_mode": "final_test",
                "test_access": "single_use_predeclared",
            }
        )
        + "\n"
    )
    score = compute_regression_metrics(
        [row["true_age"] for row in test_rows],
        [row["predicted_age"] for row in test_rows],
    )["pearsonr"]
    (run_dir / "test_completed.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evaluation_mode": "final_test",
                "test_pearsonr": score,
            }
        )
        + "\n"
    )

    audit = audit_run(run_dir)

    assert audit["status"] == "complete"
    assert audit["test_status"] == "completed"
    assert audit["test_metrics"]["pearsonr"] == pytest.approx(score)


def test_audit_keeps_official_metric_separate_from_subject_level_export(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_mode"] = "final_test"
    manifest["test_access"] = "single_use_predeclared"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    validation_rows = [
        json.loads(line)
        for line in (run_dir / "predictions" / "validation.jsonl").read_text().splitlines()
    ]
    test_rows = [dict(row, split="test") for row in validation_rows]
    (run_dir / "predictions" / "test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows)
    )
    (run_dir / "test_started.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evaluation_mode": "final_test",
                "test_access": "single_use_predeclared",
            }
        )
        + "\n"
    )
    export_score = compute_regression_metrics(
        [row["true_age"] for row in test_rows],
        [row["predicted_age"] for row in test_rows],
    )["pearsonr"]
    official_score = export_score - 0.05
    (run_dir / "test_completed.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "evaluation_mode": "final_test",
                "test_pearsonr": official_score,
                "prediction_export": {"metrics": {"pearsonr": export_score}},
            }
        )
        + "\n"
    )

    audit = audit_run(run_dir)

    assert audit["status"] == "complete"
    assert audit["official_test_pearsonr"] == pytest.approx(official_score)
    assert audit["test_metrics"]["pearsonr"] == pytest.approx(export_score)
    assert not any("test marker Pearson" in error for error in audit["errors"])


def test_audit_normalizes_external_head_parameter_contract(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    complexity_path = run_dir / "complexity.json"
    complexity = json.loads(complexity_path.read_text())
    complexity["parameter_buckets"] = {
        "encoder": {"total": 100, "trainable": 100, "frozen": 0},
        "head": {"total": 0, "trainable": 0, "frozen": 0},
        "auxiliary": [],
        "total": {"total": 100, "trainable": 100, "frozen": 0},
    }
    complexity["head_complexity"] = {"parameter_count": 10}
    complexity_path.write_text(json.dumps(complexity) + "\n")

    audit = audit_run(run_dir)

    assert audit["status"] == "complete"
    assert audit["parameter_buckets"]["head"]["total"] == 10
    assert audit["parameter_buckets"]["total"]["total"] == 110
    assert any("external head" in warning for warning in audit["warnings"])


def test_audit_uses_validation_history_for_official_selection_metric(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    selection_path = run_dir / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["selected_val_pearsonr"] = 0.75
    selection_path.write_text(json.dumps(selection) + "\n")
    history_path = run_dir / "epoch_validation_metrics.jsonl"
    history_rows = [json.loads(line) for line in history_path.read_text().splitlines()]
    history_rows[-1]["val/pearsonr"] = 0.75
    history_path.write_text("".join(json.dumps(row) + "\n" for row in history_rows))

    audit = audit_run(run_dir)

    assert audit["status"] == "complete"
    assert audit["selected_val_pearsonr"] == pytest.approx(0.75)
    assert audit["metrics"]["pearsonr"] != pytest.approx(0.75)


def test_audit_rejects_test_artifacts_in_validation_only_run(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    (run_dir / "test_started.json").write_text(
        json.dumps({"schema_version": "1.0", "evaluation_mode": "final_test"}) + "\n"
    )

    audit = audit_run(run_dir)

    assert audit["status"] == "invalid"
    assert any("validation_only" in error for error in audit["errors"])


def test_audit_rejects_selection_score_that_does_not_match_predictions(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    selection = json.loads((run_dir / "selection.json").read_text())
    selection["selected_val_pearsonr"] = 0.123
    (run_dir / "selection.json").write_text(json.dumps(selection) + "\n")

    audit = audit_run(run_dir)

    assert audit["status"] == "invalid"
    assert any("selection" in error and "Pearson" in error for error in audit["errors"])


def test_audit_rejects_malformed_selection_without_raising(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    selection = json.loads((run_dir / "selection.json").read_text())
    selection["selected_val_pearsonr"] = "not-a-number"
    (run_dir / "selection.json").write_text(json.dumps(selection) + "\n")

    audit = audit_run(run_dir)

    assert audit["status"] == "invalid"
    assert "selection.json has an invalid selected_val_pearsonr" in audit["errors"]


def test_paired_bootstrap_rejects_unmatched_subject_sets() -> None:
    baseline = {"s-1": (10.0, 10.0), "s-2": (20.0, 21.0)}
    candidate = {"s-1": (10.0, 10.0), "s-3": (30.0, 30.0)}

    with pytest.raises(ValueError, match="exactly the same subject IDs"):
        paired_bootstrap_delta(candidate, baseline, iterations=20)


def test_stability_rejects_duplicate_candidate_seed(tmp_path: Path) -> None:
    baseline = [_write_run(tmp_path, variant="mean_linear", seed=33)]
    candidate = [
        _write_run(tmp_path, variant="candidate_a", seed=33),
        _write_run(tmp_path, variant="candidate_b", seed=33),
    ]

    with pytest.raises(ValueError, match="duplicate seed"):
        summarize_seed_stability(candidate, baseline)


def test_stability_rejects_mismatched_dataset_or_split(tmp_path: Path) -> None:
    baseline = [_write_run(tmp_path, variant="mean_linear", seed=33)]
    candidate = _write_run(tmp_path, variant="candidate", seed=33)
    manifest_path = candidate / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_fingerprint"] = "different-split-sha"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="matched comparison"):
        summarize_seed_stability([candidate], baseline)


def test_stability_rejects_mismatched_comparison_config(tmp_path: Path) -> None:
    baseline = [_write_run(tmp_path, variant="mean_linear", seed=33)]
    candidate = _write_run(tmp_path, variant="candidate", seed=33)
    manifest_path = candidate / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["comparison_config_hash"] = "different-config-sha"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="comparison config"):
        summarize_seed_stability([candidate], baseline)


def test_stability_rejects_mismatched_protocol_digest(tmp_path: Path) -> None:
    baseline = [_write_run(tmp_path, variant="mean_linear", seed=33)]
    candidate = _write_run(tmp_path, variant="candidate", seed=33)
    manifest_path = candidate / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_digest"] = "different-protocol-sha"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="protocol digest"):
        summarize_seed_stability([candidate], baseline)


def test_audit_rejects_missing_reproducibility_metadata(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["source_tree_sha256"]
    manifest_path.write_text(json.dumps(manifest) + "\n")

    audit = audit_run(run_dir)

    assert audit["status"] == "invalid"
    assert any("source tree" in error for error in audit["errors"])


def test_analyze_runs_rejects_partial_evidence_package(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, variant="mean_linear", seed=33)
    (run_dir / "predictions" / "validation.jsonl").unlink()

    with pytest.raises(ValueError, match="not complete"):
        analyze_runs([run_dir], output_dir=tmp_path / "analysis")


def test_complexity_adjusted_comparison_is_null_for_nonpositive_parameter_delta(
    tmp_path: Path,
) -> None:
    baseline = _write_run(tmp_path, variant="mean_linear", seed=33)
    candidate = _write_run(tmp_path, variant="candidate", seed=33)

    comparison = complexity_adjusted_comparison([candidate], [baseline])

    assert comparison[0]["head_parameter_delta"] == 10
    assert comparison[0]["delta_per_extra_head_parameter"] is not None
