"""Prospective lock for the ds006780 capacity--data transfer experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .strict_json import canonical_sha256
from .strict_json import load_json_strict
from neurobench_age.data.ds006780 import validate_external_config


class Ds006780ExternalLockError(RuntimeError):
    """Raised when the external transfer lock is incomplete or inconsistent."""


SELECTED_HEADS = ("mean_linear", "mean_rich_stats_residual")
SELECTED_SIZES = (200, 800)
SELECTED_SEEDS = tuple(range(33, 43))
_LOCK_FIELDS = {
    "schema_version",
    "status",
    "study_id",
    "external_config_sha256",
    "target_free_manifest_sha256",
    "target_manifest_sha256",
    "participant_metadata_sha256",
    "dataset_id",
    "dataset_version",
    "source_commit",
    "checkpoint_inventory_sha256",
    "checkpoint_inventory_lock_core_sha256",
    "selected_runs_sha256",
    "selected_runs",
    "selected_run_count",
    "representation_protocol_sha256",
    "training_protocol_sha256",
    "training_source_sha256",
    "encoder_checkpoint",
    "encoder_checkpoint_sha256",
    "execution_source_sha256",
    "heads",
    "training_sizes",
    "seeds",
    "subject_ids",
    "subject_list_sha256",
    "precision_gate",
    "expected_prediction_count",
    "output_root",
    "lock_sha256",
}
_SHA_FIELDS = {
    "external_config_sha256",
    "target_free_manifest_sha256",
    "target_manifest_sha256",
    "participant_metadata_sha256",
    "checkpoint_inventory_sha256",
    "checkpoint_inventory_lock_core_sha256",
        "selected_runs_sha256",
    "representation_protocol_sha256",
    "training_protocol_sha256",
    "training_source_sha256",
    "encoder_checkpoint_sha256",
    "execution_source_sha256",
    "subject_list_sha256",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise Ds006780ExternalLockError(f"{field} must be a SHA-256 digest")
    return str(value)


def _legacy_inventory_sha256(inventory: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in inventory.items()
        if key != "checkpoint_inventory_body_sha256"
    }
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_inventory(inventory: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not isinstance(inventory, Mapping):
        raise Ds006780ExternalLockError("checkpoint inventory must be an object")
    claimed = inventory.get("checkpoint_inventory_body_sha256")
    _require_sha(claimed, "checkpoint_inventory_body_sha256")
    if claimed != _legacy_inventory_sha256(inventory):
        raise Ds006780ExternalLockError("checkpoint inventory digest is invalid")
    runs = inventory.get("runs")
    if not isinstance(runs, list) or len(runs) != 90:
        raise Ds006780ExternalLockError("capacity checkpoint inventory must contain 90 runs")
    identities: set[tuple[int, str, int]] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise Ds006780ExternalLockError(f"checkpoint inventory run {index} is invalid")
        try:
            identity = (int(run["training_size"]), str(run["head"]), int(run["seed"]))
        except (KeyError, TypeError, ValueError) as error:
            raise Ds006780ExternalLockError(
                f"checkpoint inventory run {index} has an invalid identity"
            ) from error
        if identity in identities:
            raise Ds006780ExternalLockError("checkpoint inventory contains duplicate runs")
        identities.add(identity)
        _require_sha(run.get("selected_checkpoint_sha256"), f"run[{index}].selected_checkpoint_sha256")
        _require_sha(run.get("run_manifest_sha256"), f"run[{index}].run_manifest_sha256")
    expected = {
        (size, head, seed)
        for size in (200, 400, 800)
        for head in (
            "mean_linear",
            "mean_rich_stats_residual",
            "mean_mlp_residual_matched(hidden_dim=4)",
        )
        for seed in SELECTED_SEEDS
    }
    if identities != expected:
        raise Ds006780ExternalLockError("checkpoint inventory matrix is not the complete capacity matrix")
    core_digest = inventory.get("lock_core_sha256")
    _require_sha(core_digest, "checkpoint inventory lock_core_sha256")
    return dict(inventory), str(core_digest)


def select_ds006780_runs(inventory: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Select the predeclared 40 checkpoints from the complete 90-run inventory."""

    normalized, _ = _validate_inventory(inventory)
    selected = [
        {
            "training_size": int(run["training_size"]),
            "head": str(run["head"]),
            "seed": int(run["seed"]),
            "run_manifest_sha256": str(run["run_manifest_sha256"]),
            "selected_checkpoint_sha256": str(run["selected_checkpoint_sha256"]),
            "selected_epoch": int(run["selected_epoch"]),
        }
        for run in normalized["runs"]
        if int(run["training_size"]) in SELECTED_SIZES
        and str(run["head"]) in SELECTED_HEADS
        and int(run["seed"]) in SELECTED_SEEDS
    ]
    selected.sort(key=lambda row: (row["training_size"], row["head"], row["seed"]))
    expected = {
        (size, head, seed)
        for size in SELECTED_SIZES
        for head in SELECTED_HEADS
        for seed in SELECTED_SEEDS
    }
    if {
        (row["training_size"], row["head"], row["seed"]) for row in selected
    } != expected:
        raise Ds006780ExternalLockError("selected checkpoint matrix is incomplete")
    return tuple(selected)


def _manifest_hash(manifest: Mapping[str, Any], field: str, label: str) -> str:
    claimed = manifest.get(field)
    _require_sha(claimed, f"{label}.{field}")
    body = {key: value for key, value in manifest.items() if key != field}
    if claimed != canonical_sha256(body):
        raise Ds006780ExternalLockError(f"{label}.{field} is not self-consistent")
    return str(claimed)


def _validate_target_manifests(
    target_free_manifest: Mapping[str, Any], target_manifest: Mapping[str, Any]
) -> tuple[str, str, str, tuple[str, ...]]:
    target_free_hash = _manifest_hash(target_free_manifest, "manifest_sha256", "target-free manifest")
    target_hash = _manifest_hash(target_manifest, "manifest_sha256", "target manifest")
    if target_manifest.get("target_free_manifest_sha256") != target_free_hash:
        raise Ds006780ExternalLockError("target manifest is not bound to target-free manifest")
    participant_hash = _require_sha(
        target_manifest.get("participant_metadata_sha256"),
        "target manifest participant_metadata_sha256",
    )
    candidates = target_free_manifest.get("candidate_runs")
    subjects = target_manifest.get("subjects")
    exclusions = target_manifest.get("exclusions")
    if not isinstance(candidates, list) or not isinstance(subjects, Mapping) or not isinstance(exclusions, list):
        raise Ds006780ExternalLockError("target manifests have invalid subject inventories")
    candidate_ids = [str(row.get("subject_id")) for row in candidates if isinstance(row, Mapping)]
    if len(candidate_ids) != len(candidates) or len(set(candidate_ids)) != len(candidate_ids):
        raise Ds006780ExternalLockError("target-free manifest subject inventory is invalid")
    exclusion_ids = [str(row.get("subject_id")) for row in exclusions if isinstance(row, Mapping)]
    subject_ids = [str(subject_id) for subject_id in subjects]
    if len(exclusion_ids) != len(exclusions) or len(set(exclusion_ids)) != len(exclusion_ids):
        raise Ds006780ExternalLockError("target manifest exclusions are invalid")
    if set(subject_ids) | set(exclusion_ids) != set(candidate_ids) or set(subject_ids) & set(exclusion_ids):
        raise Ds006780ExternalLockError("target manifest subject set differs from target-free manifest")
    for subject_id, record in subjects.items():
        if not isinstance(record, Mapping) or record.get("age_support_eligible") is not True:
            raise Ds006780ExternalLockError("ds006780 transfer requires in-support exact ages")
        _require_sha(record.get("signal_qc_sha256"), f"target manifest subject {subject_id} signal_qc_sha256")
    if not subject_ids:
        raise Ds006780ExternalLockError("target manifest contains no eligible subjects")
    return target_free_hash, target_hash, participant_hash, tuple(sorted(subject_ids))


def build_ds006780_external_lock(
    *,
    external_config: Mapping[str, Any],
    target_free_manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    checkpoint_inventory: Mapping[str, Any],
    selected_runs: Sequence[Mapping[str, Any]],
    execution_source_sha256: str,
    encoder_checkpoint_sha256: str,
    output_root: Path,
    checkpoint_core: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_external_config(external_config, require_approved_precision=True)
    external_config_sha256 = canonical_sha256(external_config)
    if target_free_manifest.get("protocol_sha256") != external_config_sha256:
        raise Ds006780ExternalLockError("target-free manifest protocol differs from external config")
    target_free_hash, target_hash, participant_hash, subject_ids = _validate_target_manifests(
        target_free_manifest, target_manifest
    )
    normalized_inventory, inventory_core_sha = _validate_inventory(checkpoint_inventory)
    expected_selected = select_ds006780_runs(normalized_inventory)
    selected = tuple(dict(row) for row in selected_runs)
    if tuple(selected) != expected_selected:
        raise Ds006780ExternalLockError("selected run records differ from the complete inventory")
    _require_sha(execution_source_sha256, "execution_source_sha256")
    _require_sha(encoder_checkpoint_sha256, "encoder_checkpoint_sha256")
    if checkpoint_core is None:
        raise Ds006780ExternalLockError(
            "checkpoint_core is required to bind the selected capacity checkpoints"
        )
    core = dict(checkpoint_core)
    representation_protocol_sha256 = _require_sha(
        core.get("representation_protocol_sha256"),
        "representation_protocol_sha256",
    )
    training_protocol_sha256 = _require_sha(
        core.get("training_protocol_sha256"),
        "training_protocol_sha256",
    )
    training_source_sha256 = _require_sha(
        core.get("training_source_sha256"),
        "training_source_sha256",
    )
    selected_runs_sha256 = canonical_sha256(list(selected))
    subject_list_sha256 = canonical_sha256(list(subject_ids))
    dataset = external_config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise Ds006780ExternalLockError("external config dataset contract is missing")
    body: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed",
        "study_id": "reve_age_ds006780_capacity_transfer_v1",
        "external_config_sha256": external_config_sha256,
        "target_free_manifest_sha256": target_free_hash,
        "target_manifest_sha256": target_hash,
        "participant_metadata_sha256": participant_hash,
        "dataset_id": str(dataset.get("dataset_id")),
        "dataset_version": str(dataset.get("dataset_version")),
        "source_commit": str(dataset.get("source_commit")),
        "checkpoint_inventory_sha256": str(normalized_inventory["checkpoint_inventory_body_sha256"]),
        "checkpoint_inventory_lock_core_sha256": inventory_core_sha,
        "selected_runs_sha256": selected_runs_sha256,
        "selected_runs": list(selected),
        "selected_run_count": len(selected),
        "representation_protocol_sha256": representation_protocol_sha256,
        "training_protocol_sha256": training_protocol_sha256,
        "training_source_sha256": training_source_sha256,
        "encoder_checkpoint": "brain-bzh/reve-base",
        "encoder_checkpoint_sha256": str(encoder_checkpoint_sha256),
        "execution_source_sha256": str(execution_source_sha256),
        "heads": list(SELECTED_HEADS),
        "training_sizes": list(SELECTED_SIZES),
        "seeds": list(SELECTED_SEEDS),
        "subject_ids": list(subject_ids),
        "subject_list_sha256": subject_list_sha256,
        "precision_gate": dict(external_config["precision_gate"]),
        "expected_prediction_count": len(subject_ids) * len(selected),
        "output_root": str(Path(output_root).resolve()),
    }
    return {**body, "lock_sha256": canonical_sha256(body)}


def load_ds006780_external_lock(path: Path) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, Mapping) or set(value) != _LOCK_FIELDS:
        raise Ds006780ExternalLockError("ds006780 external lock fields are invalid")
    body = {key: item for key, item in value.items() if key != "lock_sha256"}
    if value.get("status") != "sealed" or value.get("schema_version") != 1:
        raise Ds006780ExternalLockError("ds006780 external lock metadata is invalid")
    if value.get("lock_sha256") != canonical_sha256(body):
        raise Ds006780ExternalLockError("ds006780 external lock digest is invalid")
    for field in _SHA_FIELDS:
        _require_sha(value.get(field), field)
    selected = value.get("selected_runs")
    if not isinstance(selected, list) or len(selected) != 40:
        raise Ds006780ExternalLockError("ds006780 external lock must contain 40 selected runs")
    if value.get("selected_run_count") != len(selected):
        raise Ds006780ExternalLockError("selected run count is invalid")
    if value.get("selected_runs_sha256") != canonical_sha256(selected):
        raise Ds006780ExternalLockError("selected run digest is invalid")
    if value.get("heads") != list(SELECTED_HEADS) or value.get("training_sizes") != list(SELECTED_SIZES):
        raise Ds006780ExternalLockError("ds006780 external lock matrix is invalid")
    if value.get("seeds") != list(SELECTED_SEEDS):
        raise Ds006780ExternalLockError("ds006780 external lock seed inventory is invalid")
    subjects = value.get("subject_ids")
    if not isinstance(subjects, list) or len(subjects) == 0 or value.get("subject_list_sha256") != canonical_sha256(subjects):
        raise Ds006780ExternalLockError("ds006780 external lock subject inventory is invalid")
    if value.get("expected_prediction_count") != len(subjects) * 40:
        raise Ds006780ExternalLockError("ds006780 external lock prediction count is invalid")
    if not Path(str(value["output_root"])).is_absolute():
        raise Ds006780ExternalLockError("ds006780 external lock output_root must be absolute")
    return dict(value)
