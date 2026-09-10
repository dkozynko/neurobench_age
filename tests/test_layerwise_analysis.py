from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from neurobench_age.analysis.confirmatory import regression_metrics
from neurobench_age.analysis.layerwise import (
    _canonical_sha256,
    analyze_layerwise_artifacts,
    load_layerwise_analysis_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/layerwise_probe_analysis.json"


def test_layerwise_analysis_config_declares_exploratory_paired_family() -> None:
    config = load_layerwise_analysis_config(CONFIG)

    assert config.scope == "exploratory_secondary"
    assert config.baseline_head == "mean_linear_layer_m1"
    assert config.comparison_heads == (
        "mean_linear_layer_m4",
        "mean_linear_layer_m3",
        "mean_linear_layer_m2",
    )
    assert config.seeds == tuple(range(33, 43))
    assert config.bootstrap_iterations == 10_000
    assert config.bootstrap_seed == 20260910
    assert config.confidence == 0.95
    assert config.randomization_tail == "greater"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_synthetic_layerwise_evidence(root: Path) -> None:
    protocol_sha = "a" * 64
    training_sha = "b" * 64
    lock_body = {
        "schema_version": 1,
        "status": "sealed",
        "study_id": "reve_age_layerwise_probe_v1",
        "protocol_sha256": protocol_sha,
        "training_protocol_sha256": training_sha,
        "training_source_sha256": "e" * 64,
        "layer_indices": [-4, -3, -2, -1],
        "head_layers": {
            "mean_linear_layer_m4": -4,
            "mean_linear_layer_m3": -3,
            "mean_linear_layer_m2": -2,
            "mean_linear_layer_m1": -1,
        },
        "seeds": list(range(33, 43)),
        "primary_subject_count": 75,
    }
    lock = {**lock_body, "lock_sha256": _canonical_sha256(lock_body)}
    _write_json(root / "layerwise_external_lock.json", lock)

    runs: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    head_layers = lock_body["head_layers"]
    assert isinstance(head_layers, dict)
    for head, layer in head_layers.items():
        for seed in range(33, 43):
            targets = [20.0 + index * 0.4 for index in range(75)]
            predictions = [
                target
                + 0.1 * math.sin(index + seed)
                + 0.02 * abs(int(layer)) * math.cos(index / 3.0)
                for index, target in enumerate(targets)
            ]
            metrics = regression_metrics(targets, predictions)
            runs.append(
                {
                    "head_name": head,
                    "layer_index": layer,
                    "seed": seed,
                    "subject_count": 75,
                    **metrics,
                }
            )
            for index, (target, prediction) in enumerate(zip(targets, predictions)):
                body = {
                    "schema_version": 1,
                    "study_id": lock_body["study_id"],
                    "lock_sha256": lock["lock_sha256"],
                    "protocol_sha256": protocol_sha,
                    "training_protocol_sha256": training_sha,
                    "training_source_sha256": lock_body["training_source_sha256"],
                    "head_name": head,
                    "layer_index": layer,
                    "seed": seed,
                    "subject_id": f"sub-{index:03d}",
                    "true_age": target,
                    "prediction": prediction,
                }
                row = {**body, "body_sha256": _canonical_sha256(body)}
                rows.append(row)
                _write_json(
                    root
                    / "predictions"
                    / head
                    / f"seed-{seed}"
                    / f"sub-{index:03d}.json",
                    row,
                )

    metrics_body = {
        "schema_version": 1,
        "status": "complete",
        "lock_sha256": lock["lock_sha256"],
        "runs": runs,
    }
    _write_json(
        root / "external_metrics.json",
        {**metrics_body, "metrics_sha256": _canonical_sha256(metrics_body)},
    )
    inventory_body = {
        "schema_version": 1,
        "status": "complete",
        "lock_sha256": lock["lock_sha256"],
        "prediction_count": len(rows),
        "prediction_file_count": len(rows),
        "prediction_files_sha256": _canonical_sha256(
            sorted(_canonical_sha256(row) for row in rows)
        ),
    }
    _write_json(
        root / "prediction_inventory.json",
        {
            **inventory_body,
            "prediction_inventory_sha256": _canonical_sha256(inventory_body),
        },
    )


def test_layerwise_analysis_rejects_incomplete_prediction_matrix(tmp_path: Path) -> None:
    _write_synthetic_layerwise_evidence(tmp_path)
    prediction = next((tmp_path / "predictions").rglob("*.json"))
    prediction.unlink()

    config = load_layerwise_analysis_config(CONFIG)
    with pytest.raises(ValueError, match="prediction matrix"):
        analyze_layerwise_artifacts(artifact_root=tmp_path, config=config)


def test_layerwise_analysis_rejects_changed_prediction_provenance(tmp_path: Path) -> None:
    _write_synthetic_layerwise_evidence(tmp_path)
    prediction = next((tmp_path / "predictions").rglob("*.json"))
    row = json.loads(prediction.read_text(encoding="utf-8"))
    row["protocol_sha256"] = "c" * 64
    body = {key: value for key, value in row.items() if key != "body_sha256"}
    row["body_sha256"] = _canonical_sha256(body)
    prediction.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    config = load_layerwise_analysis_config(CONFIG)
    with pytest.raises(ValueError, match="provenance|digest"):
        analyze_layerwise_artifacts(artifact_root=tmp_path, config=config)


def test_layerwise_analysis_rejects_mismatched_true_age(tmp_path: Path) -> None:
    _write_synthetic_layerwise_evidence(tmp_path)
    prediction = next((tmp_path / "predictions").rglob("*.json"))
    row = json.loads(prediction.read_text(encoding="utf-8"))
    row["true_age"] = float(row["true_age"]) + 1.0
    body = {key: value for key, value in row.items() if key != "body_sha256"}
    row["body_sha256"] = _canonical_sha256(body)
    prediction.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    config = load_layerwise_analysis_config(CONFIG)
    with pytest.raises(ValueError, match="digest|target order"):
        analyze_layerwise_artifacts(artifact_root=tmp_path, config=config)


def test_layerwise_analysis_is_deterministic_and_reports_three_comparisons(
    tmp_path: Path,
) -> None:
    _write_synthetic_layerwise_evidence(tmp_path)
    config = load_layerwise_analysis_config(CONFIG)

    first = analyze_layerwise_artifacts(artifact_root=tmp_path, config=config)
    second = analyze_layerwise_artifacts(artifact_root=tmp_path, config=config)

    assert first == second
    assert first["scope"] == "exploratory_secondary"
    assert tuple(first["comparisons"]) == config.comparison_heads
    assert all(len(row["paired"]["per_seed"]) == 10 for row in first["comparisons"].values())
    assert all(row["bootstrap"]["iterations"] == 10_000 for row in first["comparisons"].values())
    assert all(row["randomization"]["permutations"] == 1024 for row in first["comparisons"].values())
    assert all("holm_adjusted_p_value" in row for row in first["comparisons"].values())
    assert len(first["analysis_sha256"]) == 64
