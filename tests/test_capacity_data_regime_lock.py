from __future__ import annotations

from pathlib import Path

import pytest

from neurobench_age.research.capacity_data_regime_lock import (
    CapacityDataRegimeLockError,
    build_checkpoint_inventory,
    build_checkpoint_sealed_lock,
    build_final_lock,
    build_lock_core,
    build_prediction_inventory,
    create_lifecycle_sidecar,
    load_checkpoint_inventory,
    load_final_lock,
    load_prediction_inventory,
    transition_lifecycle,
)


def _sha(letter: str) -> str:
    return letter * 64


def _lock_core() -> dict[str, object]:
    return {
        "extension_id": "reve_age_capacity_data_regime_v1",
        "schema_version": 1,
        "parent_primary_study_lock_sha256": _sha("a"),
        "parent_primary_prediction_inventory_sha256": _sha("b"),
        "extension_protocol_sha256": _sha("c"),
        "representation_protocol_sha256": _sha("d"),
        "training_protocol_sha256": _sha("e"),
        "training_source_sha256": _sha("f"),
        "environment_sha256": _sha("0"),
        "hardware_sha256": _sha("1"),
        "hbn_manifest_sha256": _sha("2"),
        "hbn_training_manifest_sha256": _sha("3"),
        "representation_cache_manifest_sha256": _sha("4"),
        "validation_subject_list_sha256": _sha("5"),
        "cohort_hashes": {
            "train_200": _sha("6"),
            "train_400": _sha("7"),
            "train_800": _sha("8"),
        },
        "expected_run_count": 90,
        "expected_prediction_count": 6750,
        "output_root_identity": "capacity-data-regime-output-v1",
        "preflight": {
            "representation_cache_bytes": 100,
            "estimated_extension_output_bytes": 200,
            "free_space_bytes": 1000,
            "required_free_space_bytes": 500,
            "peak_ram_bytes": 300,
            "upper_bound_optimizer_steps": 400,
            "observed_pilot_seconds": 1.25,
            "cached_window_count": 80,
        },
    }


def _runs() -> list[dict[str, object]]:
    heads = (
        "mean_linear",
        "mean_rich_stats_residual",
        "mean_mlp_residual_matched(hidden_dim=4)",
    )
    return [
        {
            "training_size": size,
            "head": head,
            "seed": seed,
            "run_manifest_sha256": _sha("a"),
            "selected_checkpoint_sha256": _sha("b"),
            "status": "complete",
            "cached_window_count": 10,
            "optimizer_steps": 40,
            "observed_early_stopping_steps": 40,
            "validation_history": [{"epoch": 1, "validation_subject_pearson": 0.1}],
            "selected_validation_subject_metrics": [
                {"subject_id": "validation", "age": 12.0, "prediction": 11.0}
            ],
            "selected_epoch": 1,
            "head_complexity": {"parameter_count": 513, "operations": ["mean_pool"]},
            "resource": {"requested_device": "cpu"},
        }
        for size in (200, 400, 800)
        for head in heads
        for seed in range(33, 43)
    ]


def _predictions() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in _runs():
        for index in range(75):
            rows.append(
                {
                    "training_size": run["training_size"],
                    "head": run["head"],
                    "seed": run["seed"],
                    "subject_id": f"sub-{index:03d}",
                    "true_age": 20.0 + index,
                    "prediction": 21.0 + index,
                    "split": "mipdb_primary",
                }
            )
    return rows


def test_lock_and_inventories_have_bidirectional_non_circular_binding() -> None:
    core = build_lock_core(_lock_core())
    checkpoint = build_checkpoint_inventory(
        core,
        runs=_runs(),
        expected_prediction_count=6750,
    )
    checkpoint_lock = build_checkpoint_sealed_lock(core, checkpoint)
    prediction = build_prediction_inventory(
        core,
        checkpoint,
        predictions=_predictions(),
    )
    final = build_final_lock(checkpoint_lock, prediction)

    assert checkpoint["lock_core_sha256"] == core["lock_core_sha256"]
    assert prediction["lock_core_sha256"] == core["lock_core_sha256"]
    assert prediction["checkpoint_inventory_body_sha256"] == checkpoint[
        "checkpoint_inventory_body_sha256"
    ]
    assert final["checkpoint_sealed_lock_sha256"] == checkpoint_lock["lock_sha256"]
    assert final["prediction_inventory_body_sha256"] == prediction[
        "prediction_inventory_body_sha256"
    ]
    assert len(final["lock_sha256"]) == 64

    load_checkpoint_inventory(checkpoint)
    load_prediction_inventory(prediction)
    load_final_lock(final)


def test_lock_rejects_core_tampering_and_missing_core_fields() -> None:
    payload = _lock_core()
    payload.pop("hardware_sha256")
    with pytest.raises(CapacityDataRegimeLockError, match="lock core fields"):
        build_lock_core(payload)

    core = build_lock_core(_lock_core())
    checkpoint = build_checkpoint_inventory(core, runs=_runs(), expected_prediction_count=6750)
    tampered = dict(checkpoint)
    tampered["lock_core_sha256"] = _sha("9")
    with pytest.raises(CapacityDataRegimeLockError, match="lock core"):
        load_checkpoint_inventory(tampered)


def test_inventory_and_final_lock_tampering_is_detected() -> None:
    core = build_lock_core(_lock_core())
    checkpoint = build_checkpoint_inventory(core, runs=_runs(), expected_prediction_count=6750)
    checkpoint_lock = build_checkpoint_sealed_lock(core, checkpoint)
    prediction = build_prediction_inventory(core, checkpoint, predictions=_predictions())
    final = build_final_lock(checkpoint_lock, prediction)

    altered_prediction = dict(prediction)
    altered_prediction["predictions"] = list(prediction["predictions"])
    altered_prediction["predictions"][0] = {
        **altered_prediction["predictions"][0],
        "prediction": 999.0,
    }
    with pytest.raises(CapacityDataRegimeLockError, match="prediction inventory"):
        load_prediction_inventory(altered_prediction)

    altered_final = dict(final)
    altered_final["prediction_inventory_body_sha256"] = _sha("9")
    with pytest.raises(CapacityDataRegimeLockError, match="final lock"):
        load_final_lock(altered_final)


def test_exact_run_matrix_and_prediction_count_are_required() -> None:
    core = build_lock_core(_lock_core())
    with pytest.raises(CapacityDataRegimeLockError, match="90"):
        build_checkpoint_inventory(
            core,
            runs=_runs()[:-1],
            expected_prediction_count=6750,
        )

    checkpoint = build_checkpoint_inventory(core, runs=_runs(), expected_prediction_count=6750)
    with pytest.raises(CapacityDataRegimeLockError, match="6,750"):
        build_prediction_inventory(core, checkpoint, predictions=_predictions()[:-1])


def test_lifecycle_is_append_only_and_bound_to_lock(tmp_path: Path) -> None:
    lifecycle_dir = tmp_path / "lifecycle"
    lock_sha = _sha("a")
    create_lifecycle_sidecar(lifecycle_dir, state="draft", lock_sha256=lock_sha)
    transition_lifecycle(
        lifecycle_dir,
        state="checkpoint_sealed",
        lock_sha256=lock_sha,
    )
    transition_lifecycle(
        lifecycle_dir,
        state="external_started",
        lock_sha256=lock_sha,
    )
    transition_lifecycle(
        lifecycle_dir,
        state="completed",
        lock_sha256=lock_sha,
    )

    with pytest.raises(CapacityDataRegimeLockError, match="immutable"):
        transition_lifecycle(
            lifecycle_dir,
            state="draft",
            lock_sha256=lock_sha,
        )
    with pytest.raises(CapacityDataRegimeLockError, match="lock"):
        transition_lifecycle(
            lifecycle_dir,
            state="failed",
            lock_sha256=_sha("b"),
        )
