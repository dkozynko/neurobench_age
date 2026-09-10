from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.research.ds006780_external_lock import (
    Ds006780ExternalLockError,
    build_ds006780_external_lock,
    load_ds006780_external_lock,
    select_ds006780_runs,
)
from neurobench_age.research.strict_json import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/research/ds006780_external_transfer.json").read_text(
        encoding="utf-8"
    )
)


def _inventory() -> dict[str, object]:
    runs = []
    for size in (200, 400, 800):
        for head in (
            "mean_linear",
            "mean_rich_stats_residual",
            "mean_mlp_residual_matched(hidden_dim=4)",
        ):
            for seed in range(33, 43):
                runs.append(
                    {
                        "training_size": size,
                        "head": head,
                        "seed": seed,
                        "run_manifest_sha256": f"{size:03d}{seed:02d}".ljust(64, "a"),
                        "selected_checkpoint_sha256": f"{size:03d}{seed:02d}".ljust(64, "b"),
                        "selected_epoch": 4,
                    }
                )
    body = {
        "schema_version": 1,
        "status": "complete",
        "lock_core_sha256": "c" * 64,
        "run_count": len(runs),
        "expected_prediction_count": 6750,
        "runs": runs,
    }
    return {**body, "checkpoint_inventory_body_sha256": canonical_sha256(body)}


def _manifests() -> tuple[dict[str, object], dict[str, object]]:
    protocol_sha = canonical_sha256(CONFIG)
    target_free = {
        "schema_version": 1,
        "dataset_id": "ds006780",
        "protocol_sha256": protocol_sha,
        "candidate_runs": [
            {"subject_id": "sub-001", "run_id": "run-01"},
            {"subject_id": "sub-002", "run_id": "run-01"},
        ],
    }
    target_free["manifest_sha256"] = canonical_sha256(target_free)
    target = {
        "schema_version": 1,
        "target_free_manifest_sha256": target_free["manifest_sha256"],
        "participant_metadata_sha256": "2" * 64,
        "subjects": {
            "sub-001": {
                "signal_qc_sha256": "3" * 64,
                "age_years": 10.0,
                "age_units": "years",
                "age_support_eligible": True,
            },
            "sub-002": {
                "signal_qc_sha256": "4" * 64,
                "age_years": 12.0,
                "age_units": "years",
                "age_support_eligible": True,
            },
        },
        "exclusions": [],
    }
    target["manifest_sha256"] = canonical_sha256(target)
    return target_free, target


def test_select_ds006780_runs_returns_the_declared_40_run_matrix() -> None:
    selected = select_ds006780_runs(_inventory())

    assert len(selected) == 40
    assert {(row["training_size"], row["head"], row["seed"]) for row in selected} == {
        (size, head, seed)
        for size in (200, 800)
        for head in ("mean_linear", "mean_rich_stats_residual")
        for seed in range(33, 43)
    }


def test_build_and_load_ds006780_external_lock_binds_target_and_checkpoint_identity(
    tmp_path: Path,
) -> None:
    target_free, target = _manifests()
    selected = select_ds006780_runs(_inventory())
    lock = build_ds006780_external_lock(
        external_config=CONFIG,
        target_free_manifest=target_free,
        target_manifest=target,
        checkpoint_inventory=_inventory(),
        selected_runs=selected,
        checkpoint_core={
            "representation_protocol_sha256": "8" * 64,
            "training_protocol_sha256": "9" * 64,
            "training_source_sha256": "a" * 64,
        },
        execution_source_sha256="6" * 64,
        encoder_checkpoint_sha256="7" * 64,
        output_root=tmp_path / "outputs",
    )

    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    loaded = load_ds006780_external_lock(path)

    assert loaded["status"] == "sealed"
    assert loaded["selected_run_count"] == 40
    assert loaded["expected_prediction_count"] == 80
    assert loaded["target_manifest_sha256"] == target["manifest_sha256"]


def test_build_lock_rejects_subject_not_in_target_free_manifest() -> None:
    target_free, target = _manifests()
    target["subjects"]["sub-999"] = target["subjects"].pop("sub-002")
    target["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in target.items() if key != "manifest_sha256"}
    )

    with pytest.raises(Ds006780ExternalLockError, match="target-free manifest"):
        build_ds006780_external_lock(
            external_config=CONFIG,
            target_free_manifest=target_free,
            target_manifest=target,
            checkpoint_inventory=_inventory(),
            selected_runs=select_ds006780_runs(_inventory()),
            checkpoint_core={
                "representation_protocol_sha256": "8" * 64,
                "training_protocol_sha256": "9" * 64,
                "training_source_sha256": "a" * 64,
            },
            execution_source_sha256="6" * 64,
            encoder_checkpoint_sha256="7" * 64,
            output_root=Path("/tmp/ds006780-lock-test"),
        )
