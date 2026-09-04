from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.analysis.gate import GateError, evaluate_validation_gate


def _write_run(
    root: Path,
    *,
    seed: int,
    score: float,
    evaluation_mode: str = "validation_only",
    test_status: str = "withheld",
) -> Path:
    run = root / f"seed{seed}"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "completed",
                "seed": seed,
                "task": "age",
                "dataset_manifest": "dataset-sha",
                "split_fingerprint": "split-sha",
                "source_tree_sha256": "source-sha",
                "protocol_digest": "protocol-sha",
                "comparison_config_hash": "comparison-sha",
                "comparison_factor_keys": ["head_variant"],
                "encoder_checkpoint": "brain-bzh/reve-base",
                "deterministic_policy": "strict",
                "deterministic_policy_satisfied": True,
                "deterministic_settings": {
                    "algorithms": True,
                    "cudnn_benchmark": False,
                },
                "evaluation_mode": evaluation_mode,
                "test_access": "sealed",
            }
        )
    )
    (run / "selection.json").write_text(json.dumps({"selected_val_pearsonr": score}))
    (run / "report.json").write_text(json.dumps({"test_status": test_status}))
    prediction_dir = run / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "validation.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "subject_id": subject_id,
                    "true_age": true_age,
                    "predicted_age": true_age + 0.5,
                    "split": "validation",
                    "seed": seed,
                }
            )
            + "\n"
            for subject_id, true_age in (("s-1", 10.0), ("s-2", 20.0))
        )
    )
    return run


def test_validation_gate_pairs_by_seed_and_reports_wins(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline_runs = [_write_run(baseline, seed=33, score=0.70), _write_run(baseline, seed=34, score=0.71)]
    candidate_runs = [_write_run(candidate, seed=34, score=0.73), _write_run(candidate, seed=33, score=0.72)]

    result = evaluate_validation_gate(baseline_runs, candidate_runs, minimum_wins=2)

    assert result["status"] == "passed"
    assert result["wins"] == 2
    assert [row["seed"] for row in result["per_seed"]] == [33, 34]


def test_validation_gate_rejects_test_contamination(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", seed=33, score=0.70)
    candidate = _write_run(
        tmp_path / "candidate",
        seed=33,
        score=0.72,
        evaluation_mode="final_test",
        test_status="completed",
    )

    with pytest.raises(GateError, match="validation-only"):
        evaluate_validation_gate([baseline], [candidate])


def test_validation_gate_rejects_mismatched_provenance(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", seed=33, score=0.70)
    candidate = _write_run(tmp_path / "candidate", seed=33, score=0.72)
    manifest_path = candidate / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_fingerprint"] = "different-split"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(GateError, match="split_fingerprint"):
        evaluate_validation_gate([baseline], [candidate])
