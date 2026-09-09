from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neurobench_age.pipelines.capacity_data_regime import _head_directory_name
from neurobench_age.pipelines.capacity_data_regime_external import (
    CapacityDataRegimeExternalError,
    run_capacity_data_regime_external,
)
from neurobench_age.research.capacity_data_regime_lock import (
    build_checkpoint_inventory,
    build_checkpoint_sealed_lock,
    build_lock_core,
    load_checkpoint_sealed_lock,
)
from neurobench_age.heads.math import MeanLinearCopyHead


def test_external_cli_exposes_lock_and_secondary_boundary_arguments() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_capacity_data_regime_external.py"
    spec = importlib.util.spec_from_file_location("capacity_external_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = {action.dest for action in module.build_parser()._actions}
    assert {
        "checkpoint_sealed_lock",
        "checkpoint_inventory",
        "mipdb_subjects",
        "representation_root",
        "output_root",
    }.issubset(actions)


def _sha(value: str) -> str:
    return value * 64


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core(primary_lock: Path, primary_inventory: Path) -> dict[str, object]:
    return {
        "extension_id": "reve_age_capacity_data_regime_v1",
        "schema_version": 1,
        "parent_primary_study_lock_sha256": _file_sha(primary_lock),
        "parent_primary_prediction_inventory_sha256": _file_sha(primary_inventory),
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
        "output_root_identity": "external-test",
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


def _fixture(tmp_path: Path) -> dict[str, object]:
    primary_lock = tmp_path / "primary-study-lock.json"
    primary_inventory = tmp_path / "primary-prediction-inventory.json"
    primary_lock.write_text("primary lock\n", encoding="utf-8")
    primary_inventory.write_text("primary inventory\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoints"
    runs: list[dict[str, object]] = []
    for size in (200, 400, 800):
        for head in (
            "mean_linear",
            "mean_rich_stats_residual",
            "mean_mlp_residual_matched(hidden_dim=4)",
        ):
            for seed in range(33, 43):
                run_dir = checkpoint_root / f"n-{size}" / _head_directory_name(head) / f"seed-{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                model = MeanLinearCopyHead(embed_dim=2, n_outputs=1)
                if head == "mean_rich_stats_residual":
                    from neurobench_age.heads.math import MeanRichStatsResidualHead

                    model = MeanRichStatsResidualHead(embed_dim=2, n_outputs=1)
                elif "mlp" in head:
                    from neurobench_age.heads.math import MeanMLPResidualHead

                    model = MeanMLPResidualHead(embed_dim=2, n_outputs=1, hidden_dim=4)
                checkpoint = run_dir / "head_checkpoint.pt"
                torch.save(
                    {
                        "schema_version": 3,
                        "state_dict": model.state_dict(),
                        "head_name": head,
                        "seed": seed,
                        "selected_epoch": 1,
                        "representation_protocol_sha256": _sha("d"),
                        "training_protocol_sha256": _sha("e"),
                        "training_source_sha256": _sha("f"),
                        "run_identity_sha256": _sha("a"),
                    },
                    checkpoint,
                )
                runs.append(
                    {
                        "training_size": size,
                        "head": head,
                        "seed": seed,
                        "run_manifest_sha256": _sha("b"),
                        "selected_checkpoint_sha256": _file_sha(checkpoint),
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
                )
    core_bundle = build_lock_core(_core(primary_lock, primary_inventory))
    checkpoint_inventory = build_checkpoint_inventory(
        core_bundle, runs=runs, expected_prediction_count=6750
    )
    checkpoint_lock = build_checkpoint_sealed_lock(core_bundle, checkpoint_inventory)
    lock_path = tmp_path / "checkpoint_sealed_lock.json"
    inventory_path = tmp_path / "checkpoint_inventory.json"
    lock_path.write_text(json.dumps(checkpoint_lock, sort_keys=True), encoding="utf-8")
    inventory_path.write_text(json.dumps(checkpoint_inventory, sort_keys=True), encoding="utf-8")
    (tmp_path / "lifecycle").mkdir()
    (tmp_path / "lifecycle/lifecycle-checkpoint_sealed.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "checkpoint_sealed",
                "lock_sha256": checkpoint_lock["lock_core_sha256"],
            }
        ),
        encoding="utf-8",
    )
    subjects = tuple(
        {"subject_id": f"mipdb-{index:03d}", "true_age": 20.0 + index}
        for index in range(75)
    )
    return {
        "primary_lock": primary_lock,
        "primary_inventory": primary_inventory,
        "checkpoint_root": checkpoint_root,
        "lock_path": lock_path,
        "inventory_path": inventory_path,
        "subjects": subjects,
    }


def test_external_evaluator_creates_6750_predictions_and_final_lock(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def provider(subject_id: str) -> torch.Tensor:
        index = int(subject_id.rsplit("-", 1)[1])
        return torch.full((3, 2, 2), 1.0 + index / 10.0)

    result = run_capacity_data_regime_external(
        checkpoint_sealed_lock_path=fixture["lock_path"],
        checkpoint_inventory_path=fixture["inventory_path"],
        checkpoint_root=fixture["checkpoint_root"],
        primary_study_lock=fixture["primary_lock"],
        primary_prediction_inventory=fixture["primary_inventory"],
        subjects=fixture["subjects"],
        output_root=tmp_path / "external-output",
        representation_provider=provider,
    )

    assert result["status"] == "completed"
    assert result["prediction_inventory"]["prediction_count"] == 6750
    assert (tmp_path / "external-output/final_lock.json").is_file()
    assert len(result["external_metrics"]["runs"]) == 90
    resumed = run_capacity_data_regime_external(
        checkpoint_sealed_lock_path=fixture["lock_path"],
        checkpoint_inventory_path=fixture["inventory_path"],
        checkpoint_root=fixture["checkpoint_root"],
        primary_study_lock=fixture["primary_lock"],
        primary_prediction_inventory=fixture["primary_inventory"],
        subjects=fixture["subjects"],
        output_root=tmp_path / "external-output",
        representation_provider=provider,
    )
    assert resumed["final_lock"] == result["final_lock"]


def test_external_evaluator_rejects_non_sealed_parent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    lock = json.loads(Path(fixture["lock_path"]).read_text(encoding="utf-8"))
    lock["status"] = "final"
    with pytest.raises(Exception, match="checkpoint-sealed lock"):
        load_checkpoint_sealed_lock(lock)


def test_external_evaluator_rechecks_primary_snapshot_before_materialization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    changed = False

    def provider(subject_id: str) -> torch.Tensor:
        nonlocal changed
        if not changed:
            Path(fixture["primary_inventory"]).write_text("tampered\n", encoding="utf-8")
            changed = True
        index = int(subject_id.rsplit("-", 1)[1])
        return torch.full((3, 2, 2), 1.0 + index / 10.0)

    with pytest.raises(CapacityDataRegimeExternalError, match="primary evidence changed"):
        run_capacity_data_regime_external(
            checkpoint_sealed_lock_path=fixture["lock_path"],
            checkpoint_inventory_path=fixture["inventory_path"],
            checkpoint_root=fixture["checkpoint_root"],
            primary_study_lock=fixture["primary_lock"],
            primary_prediction_inventory=fixture["primary_inventory"],
            subjects=fixture["subjects"],
            output_root=tmp_path / "external-output",
            representation_provider=provider,
        )
