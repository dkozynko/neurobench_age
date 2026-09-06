"""Artifact-derived validation for publishing a prospective study lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.data.medium_subset import manifest_sha256
from neurobench_age.data.mipdb import (
    MipdbInventoryError,
    verify_mipdb_acquisition,
)
from neurobench_age.pipelines.frozen_probe import RepresentationCacheIdentity
from neurobench_age.pipelines.frozen_probe_training import (
    APPROVED_HEADS,
    FrozenEncoderError,
    load_frozen_probe_training_manifest,
)
from neurobench_age.pipelines.representation_materialization import (
    preprocessing_contract_sha256,
)
from neurobench_age.research.protocol import StudyProtocol

from .study_lock import (
    StudyLockError,
    canonical_sha256,
    load_checkpoint_inventory,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyLockError(f"could not read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise StudyLockError(f"{label} must contain a JSON object")
    return payload


def verify_checkpoint_artifacts(
    *, checkpoint_root: Path, inventory_path: Path
) -> dict[str, Any]:
    """Verify every inventory record against its run manifest and checkpoint bytes."""

    inventory = load_checkpoint_inventory(Path(inventory_path))
    root = Path(checkpoint_root)
    expected_run_roots = {
        root / record["head_name"] / f"seed-{record['seed']}"
        for record in inventory["runs"]
    }
    actual_run_roots = {
        run_root
        for head_root in root.iterdir()
        if head_root.is_dir()
        for run_root in head_root.iterdir()
        if run_root.is_dir()
    } if root.is_dir() else set()
    if actual_run_roots != expected_run_roots:
        raise StudyLockError("checkpoint run directory inventory is not exact")
    for record in inventory["runs"]:
        head_name = record["head_name"]
        seed = record["seed"]
        run_root = root / head_name / f"seed-{seed}"
        manifest_path = run_root / "run_manifest.json"
        checkpoint_path = run_root / "head_checkpoint.pt"
        if {path.name for path in run_root.iterdir()} != {
            "run_manifest.json",
            "head_checkpoint.pt",
        }:
            raise StudyLockError(f"checkpoint run file inventory is not exact: {run_root}")
        if not manifest_path.is_file():
            raise StudyLockError(f"checkpoint run manifest is missing: {manifest_path}")
        if not checkpoint_path.is_file():
            raise StudyLockError(f"checkpoint is missing: {checkpoint_path}")
        manifest = _load_json(manifest_path, "checkpoint run manifest")
        claimed_manifest_sha256 = manifest.get("run_manifest_sha256")
        manifest_body = {
            key: value
            for key, value in manifest.items()
            if key != "run_manifest_sha256"
        }
        if (
            claimed_manifest_sha256 != canonical_sha256(manifest_body)
            or claimed_manifest_sha256 != record["run_manifest_sha256"]
        ):
            raise StudyLockError(f"checkpoint run manifest hash differs: {manifest_path}")
        head_parameters = manifest.get("head_parameters")
        if (
            manifest.get("status") != "complete"
            or manifest.get("head_name") != head_name
            or manifest.get("seed") != seed
            or manifest.get("run_identity_sha256") != record["run_identity_sha256"]
            or manifest.get("selected_epoch") != record["selected_epoch"]
            or manifest.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or not isinstance(head_parameters, Mapping)
            or head_parameters.get("trainable")
            != record["head_parameter_count"]
        ):
            raise StudyLockError(f"checkpoint run identity differs: {run_root}")
        if _sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
            raise StudyLockError(f"checkpoint hash differs: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise StudyLockError(f"checkpoint is unreadable: {checkpoint_path}") from error
        if (
            not isinstance(checkpoint, dict)
            or not isinstance(checkpoint.get("state_dict"), dict)
            or checkpoint.get("head_name") != head_name
            or checkpoint.get("seed") != seed
            or checkpoint.get("selected_epoch") != record["selected_epoch"]
            or checkpoint.get("run_identity_sha256") != record["run_identity_sha256"]
        ):
            raise StudyLockError(f"checkpoint payload identity differs: {checkpoint_path}")
    return inventory


def _validate_hbn_acquisition(
    *, training_manifest: Mapping[str, Any], hbn_data_root: Path
) -> None:
    files = training_manifest.get("acquisition_files")
    if not isinstance(files, list) or not files:
        raise StudyLockError("HBN training acquisition inventory is missing")
    root = Path(hbn_data_root).resolve()
    declared_paths: set[Path] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "subject_id",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise StudyLockError("HBN training acquisition record is invalid")
        path = (root / str(record["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise StudyLockError("HBN acquisition path escapes its root") from error
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise StudyLockError(f"HBN acquisition differs from training evidence: {path}")
        if path in declared_paths:
            raise StudyLockError("HBN acquisition inventory contains duplicate paths")
        declared_paths.add(path)
    for path in tuple(declared_paths):
        if path.suffix.casefold() == ".set":
            companion = path.with_suffix(".fdt")
            if companion.is_file() and companion not in declared_paths:
                raise StudyLockError(
                    f"HBN acquisition companion is absent from training evidence: {companion}"
                )
    expected_dataset_sha256 = canonical_sha256(
        {
            "subject_manifest_sha256": training_manifest.get(
                "subject_manifest_sha256"
            ),
            "acquisition_files": files,
        }
    )
    if training_manifest.get("dataset_manifest_sha256") != expected_dataset_sha256:
        raise StudyLockError("HBN dataset identity differs from training evidence")


def _validate_checkpoint_contracts(
    *,
    checkpoint_root: Path,
    inventory: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    protocol: StudyProtocol,
) -> None:
    cache_contract = {
        field: training_manifest[field]
        for field in (
            "protocol_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "dataset_manifest_sha256",
            "preprocessing_sha256",
            "source_tree_sha256",
        )
    }
    training_contract = {
        "seeds": list(protocol.training.seeds),
        "optimizer": protocol.training.optimizer,
        "learning_rate": protocol.training.learning_rate,
        "weight_decay": protocol.training.weight_decay,
        "batch_size": protocol.training.batch_size,
        "max_epochs": protocol.training.max_epochs,
        "patience": protocol.training.patience,
        "loss": protocol.training.loss,
        "checkpoint_metric": protocol.training.checkpoint_metric,
        "metric_mode": protocol.training.metric_mode,
    }
    expected_subjects = []
    for subject in training_manifest["subjects"]:
        identity = RepresentationCacheIdentity(
            protocol_sha256=cache_contract["protocol_sha256"],
            checkpoint=cache_contract["checkpoint"],
            checkpoint_sha256=cache_contract["checkpoint_sha256"],
            dataset_manifest_sha256=cache_contract["dataset_manifest_sha256"],
            preprocessing_sha256=cache_contract["preprocessing_sha256"],
            subject_id=subject["subject_id"],
            source_tree_sha256=cache_contract["source_tree_sha256"],
        )
        expected_subjects.append(
            {
                "subject_id": subject["subject_id"],
                "split": subject["split"],
                "age": float(subject["age"]),
                "cache_key": identity.key,
            }
        )
    for record in inventory["runs"]:
        path = (
            Path(checkpoint_root)
            / record["head_name"]
            / f"seed-{record['seed']}"
            / "run_manifest.json"
        )
        manifest = _load_json(path, "checkpoint run manifest")
        identity = manifest.get("run_identity")
        expected_identity = {
            "schema_version": 1,
            "head_name": record["head_name"],
            "seed": record["seed"],
            "cache_contract": cache_contract,
            "training": training_contract,
            "subjects": expected_subjects,
        }
        if identity != expected_identity or canonical_sha256(expected_identity) != record[
            "run_identity_sha256"
        ]:
            raise StudyLockError(f"checkpoint training contract differs: {path}")


def _repository_provenance(repository_root: Path) -> tuple[str, str, bool]:
    root = Path(repository_root).resolve()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise StudyLockError("could not establish repository provenance") from error
    if not revision:
        raise StudyLockError("Git revision is empty")
    if dirty:
        raise StudyLockError("study sealing requires a clean Git worktree")
    return source_tree_sha256(root), revision, dirty


def derive_study_payload(
    *,
    protocol: StudyProtocol,
    repository_root: Path,
    environment_path: Path,
    hbn_subject_manifest_path: Path,
    hbn_training_manifest_path: Path,
    hbn_data_root: Path,
    checkpoint_root: Path,
    checkpoint_inventory_path: Path,
    mipdb_manifest_path: Path,
    mipdb_bids_root: Path,
    mipdb_pilot_qc_path: Path,
    mipdb_cohort_qc_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Derive a strict lock payload from mutually verified study artifacts."""

    source_sha256, git_revision, git_dirty = _repository_provenance(repository_root)
    environment_path = Path(environment_path)
    if not environment_path.is_file():
        raise StudyLockError("environment lock is missing")

    training_manifest = _load_json(hbn_training_manifest_path, "HBN training manifest")
    try:
        records = load_frozen_probe_training_manifest(
            Path(hbn_training_manifest_path), protocol=protocol
        )
    except (FrozenEncoderError, OSError, ValueError) as error:
        raise StudyLockError(str(error)) from error
    hbn_manifest_sha256 = manifest_sha256(Path(hbn_subject_manifest_path))
    if training_manifest.get("subject_manifest_sha256") != hbn_manifest_sha256:
        raise StudyLockError("HBN source manifest differs from training evidence")
    if training_manifest.get("source_tree_sha256") != source_sha256:
        raise StudyLockError("HBN training source differs from the sealing source")
    expected_preprocessing_sha256 = preprocessing_contract_sha256(
        protocol.preprocessing
    )
    if training_manifest.get("preprocessing_sha256") != expected_preprocessing_sha256:
        raise StudyLockError("HBN preprocessing differs from the protocol")
    _validate_hbn_acquisition(
        training_manifest=training_manifest, hbn_data_root=hbn_data_root
    )

    inventory = verify_checkpoint_artifacts(
        checkpoint_root=checkpoint_root,
        inventory_path=checkpoint_inventory_path,
    )
    _validate_checkpoint_contracts(
        checkpoint_root=checkpoint_root,
        inventory=inventory,
        training_manifest=training_manifest,
        protocol=protocol,
    )

    mipdb_manifest = _load_json(mipdb_manifest_path, "finalized MIPDB manifest")
    if (
        mipdb_manifest.get("schema_version") != 2
        or mipdb_manifest.get("status") != "finalized"
        or mipdb_manifest.get("protocol_sha256") != protocol.sha256
    ):
        raise StudyLockError("MIPDB manifest is not finalized for this protocol")
    try:
        verify_mipdb_acquisition(Path(mipdb_bids_root), mipdb_manifest)
    except MipdbInventoryError as error:
        raise StudyLockError(str(error)) from error
    cohorts = mipdb_manifest.get("cohorts")
    subject_hashes = mipdb_manifest.get("subject_list_sha256")
    if not isinstance(cohorts, dict) or not isinstance(subject_hashes, dict):
        raise StudyLockError("MIPDB finalized cohort evidence is invalid")
    for name in ("pilot", "primary", "extrapolation"):
        if not isinstance(cohorts.get(name), list) or subject_hashes.get(
            name
        ) != canonical_sha256(cohorts[name]):
            raise StudyLockError(f"MIPDB {name} cohort hash differs")
    underpowered = len(cohorts["primary"]) < protocol.datasets.minimum_primary_subjects
    if (
        mipdb_manifest.get("minimum_primary_subjects")
        != protocol.datasets.minimum_primary_subjects
        or mipdb_manifest.get("underpowered") is not underpowered
    ):
        raise StudyLockError("MIPDB finalized power evidence differs")

    pilot = _load_json(mipdb_pilot_qc_path, "MIPDB pilot QC")
    if (
        pilot.get("status") != "passed"
        or pilot.get("protocol_sha256") != protocol.sha256
        or pilot.get("dataset_manifest_sha256")
        != mipdb_manifest.get("dataset_manifest_sha256")
        or pilot.get("draft_manifest_sha256")
        != mipdb_manifest.get("draft_manifest_sha256")
        or pilot.get("pilot_subject_list_sha256") != subject_hashes["pilot"]
        or pilot.get("pilot_subject_count") != protocol.datasets.pilot_size
        or pilot.get("checkpoint") != protocol.encoder.checkpoint
        or pilot.get("preprocessing_sha256") != expected_preprocessing_sha256
    ):
        raise StudyLockError("MIPDB pilot QC differs from finalized study artifacts")
    cohort_qc = _load_json(mipdb_cohort_qc_path, "MIPDB cohort QC")
    cohort_qc_body = {
        key: value for key, value in cohort_qc.items() if key != "cohort_qc_sha256"
    }
    if (
        cohort_qc.get("status") != "complete"
        or cohort_qc.get("protocol_sha256") != protocol.sha256
        or cohort_qc.get("dataset_manifest_sha256")
        != mipdb_manifest.get("dataset_manifest_sha256")
        or cohort_qc.get("cohort_qc_sha256") != canonical_sha256(cohort_qc_body)
        or cohort_qc.get("cohort_qc_sha256")
        != mipdb_manifest.get("cohort_qc_sha256")
        or cohort_qc.get("draft_manifest_sha256")
        != mipdb_manifest.get("draft_manifest_sha256")
    ):
        raise StudyLockError("MIPDB cohort QC differs from finalized manifest")
    encoder_sha256 = training_manifest.get("checkpoint_sha256")
    if pilot.get("checkpoint_state_sha256") != encoder_sha256:
        raise StudyLockError("pilot and HBN encoder state hashes differ")

    hbn_train = [record.subject_id for record in records if record.split == "train"]
    hbn_validation = [
        record.subject_id for record in records if record.split == "validation"
    ]
    external_output = Path(output_root)
    if not external_output.is_absolute():
        raise StudyLockError("external output root must be absolute")
    return {
        "study_id": protocol.study_id,
        "protocol_sha256": protocol.sha256,
        "source_tree_sha256": source_sha256,
        "git_revision": git_revision,
        "git_dirty": git_dirty,
        "encoder_checkpoint": protocol.encoder.checkpoint,
        "encoder_checkpoint_sha256": encoder_sha256,
        "checkpoint_inventory_sha256": inventory[
            "checkpoint_inventory_sha256"
        ],
        "environment_sha256": _sha256_file(environment_path),
        "hbn_manifest_sha256": hbn_manifest_sha256,
        "hbn_training_manifest_sha256": _sha256_file(hbn_training_manifest_path),
        "mipdb_manifest_sha256": _sha256_file(mipdb_manifest_path),
        "mipdb_pilot_qc_sha256": _sha256_file(mipdb_pilot_qc_path),
        "mipdb_cohort_qc_sha256": _sha256_file(mipdb_cohort_qc_path),
        "subject_list_sha256": {
            "hbn_train": canonical_sha256(hbn_train),
            "hbn_validation": canonical_sha256(hbn_validation),
            "mipdb_pilot": subject_hashes["pilot"],
            "mipdb_primary": subject_hashes["primary"],
            "mipdb_extrapolation": subject_hashes["extrapolation"],
        },
        "heads": list(APPROVED_HEADS),
        "seeds": list(protocol.training.seeds),
        "preprocessing_sha256": expected_preprocessing_sha256,
        "statistics_sha256": protocol.statistics_sha256,
        "output_root": str(external_output.resolve()),
    }
