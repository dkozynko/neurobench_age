from __future__ import annotations

from neurobench_age.analysis.capacity_data_regime import analyze_capacity_data_regime
from neurobench_age.research.capacity_data_regime_lock import (
    build_checkpoint_inventory,
    build_checkpoint_sealed_lock,
    build_final_lock,
    build_lock_core,
    build_prediction_inventory,
)


def _sha(letter: str) -> str:
    return letter * 64


def _evidence() -> tuple[dict[str, object], dict[str, object]]:
    core = build_lock_core(
        {
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
            "output_root_identity": "analysis-test",
            "preflight": {
                "representation_cache_bytes": 100,
                "estimated_extension_output_bytes": 200,
                "free_space_bytes": 1000,
                "required_free_space_bytes": 500,
                "peak_ram_bytes": 300,
                "upper_bound_optimizer_steps": 400,
                "observed_pilot_seconds": 1.0,
                "cached_window_count": 80,
            },
        }
    )
    heads = (
        "mean_linear",
        "mean_rich_stats_residual",
        "mean_mlp_residual_matched(hidden_dim=4)",
    )
    runs = [
        {
            "training_size": size,
            "head": head,
            "seed": seed,
            "run_manifest_sha256": _sha("9"),
            "selected_checkpoint_sha256": _sha("0"),
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
    checkpoint = build_checkpoint_inventory(core, runs=runs, expected_prediction_count=6750)
    predictions: list[dict[str, object]] = []
    for run in runs:
        size = int(run["training_size"])
        head = str(run["head"])
        seed = int(run["seed"])
        for subject_index in range(75):
            age = 20.0 + subject_index
            baseline = age + 0.02 * (seed - 37) + 0.001 * size
            if head == "mean_linear":
                prediction = baseline
            elif head == "mean_rich_stats_residual":
                prediction = baseline + 0.03 * (size / 200.0) * ((subject_index % 5) - 2)
            else:
                prediction = baseline + 0.05 * (size / 200.0) * ((subject_index % 7) - 3)
            predictions.append(
                {
                    "training_size": size,
                    "head": head,
                    "seed": seed,
                    "subject_id": f"mipdb-{subject_index:03d}",
                    "true_age": age,
                    "prediction": prediction,
                    "split": "mipdb_primary",
                }
            )
    prediction_inventory = build_prediction_inventory(
        core, checkpoint, predictions=predictions
    )
    return (
        build_final_lock(build_checkpoint_sealed_lock(core, checkpoint), prediction_inventory),
        prediction_inventory,
        checkpoint,
    )


def test_analysis_reports_exact_cells_contrasts_and_joint_bootstrap() -> None:
    final_lock, prediction_inventory, checkpoint_inventory = _evidence()
    first = analyze_capacity_data_regime(
        final_lock=final_lock,
        prediction_inventory=prediction_inventory,
        checkpoint_inventory=checkpoint_inventory,
    )
    second = analyze_capacity_data_regime(
        final_lock=final_lock,
        prediction_inventory=prediction_inventory,
        checkpoint_inventory=checkpoint_inventory,
    )

    assert first == second
    assert len(first["cells"]) == 6
    assert len(first["contrasts"]) == 6
    assert first["bootstrap"]["iterations"] == 10_000
    assert first["bootstrap"]["seed"] == 20260909
    assert all(
        result["bootstrap"]["valid_iterations"] == 10_000
        for result in first["contrasts"]
    )
    assert all(
        result["bootstrap"]["shared_seed_and_subject_draws"] is True
        for result in first["contrasts"]
    )
