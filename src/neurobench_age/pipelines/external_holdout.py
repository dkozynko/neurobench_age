"""One-time, append-only inference over the sealed external MIPDB cohort."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from neurobench_age.research.study_lock import (
    StudyLockError,
    canonical_sha256,
    load_checkpoint_inventory,
    load_study_lock,
    transition_study,
)

from .frozen_probe import (
    PREDECLARED_LAYERS,
    FrozenEncoderError,
    RepresentationCacheIdentity,
    load_cached_representations,
)
from .frozen_probe_training import (
    APPROVED_HEADS,
    build_frozen_probe_head,
    required_layer_for_head,
)


INFERENCE_BATCH_SIZE = 64
_SAFE_SUBJECT_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class ExternalHoldoutError(RuntimeError):
    """Raised before unsafe or non-reproducible holdout access."""


@dataclass(frozen=True)
class RuntimeProvenance:
    training_source_sha256: str
    git_revision: str
    git_dirty: bool
    environment_sha256: str


@dataclass(frozen=True)
class ExternalSubject:
    subject_id: str
    age: float


@dataclass(frozen=True)
class ExternalSubjectMaterial:
    representations: Mapping[int, torch.Tensor]
    cache_identity: RepresentationCacheIdentity
    qc: Mapping[str, Any]


@dataclass(frozen=True)
class _LoadedHead:
    head_name: str
    seed: int
    checkpoint_sha256: str
    model: nn.Module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalHoldoutError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ExternalHoldoutError(f"{description} must contain a JSON object")
    return value


def _publish_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise ExternalHoldoutError(f"immutable artifact already exists: {path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_state(lock_path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    state = _load_json(lock_path.parent / "study_state.json", "study state")
    if state.get("lock_sha256") != lock["lock_sha256"]:
        raise ExternalHoldoutError("study state does not belong to the sealed lock")
    if state.get("state") not in {"sealed", "started"}:
        raise ExternalHoldoutError("external holdout requires a sealed or started study")
    return state


def _validate_runtime(
    lock: Mapping[str, Any],
    runtime: RuntimeProvenance,
    environment_path: Path,
) -> None:
    actual_environment_sha256 = _sha256_file(environment_path)
    expected = {
        "training_source_sha256": runtime.training_source_sha256,
        "git_revision": runtime.git_revision,
        "git_dirty": runtime.git_dirty,
        "environment_sha256": runtime.environment_sha256,
    }
    mismatches = [
        name for name, value in expected.items() if lock.get(name) != value
    ]
    if actual_environment_sha256 != runtime.environment_sha256:
        mismatches.append("environment_file_sha256")
    if mismatches:
        raise ExternalHoldoutError(
            "runtime provenance does not match sealed study: "
            + ", ".join(sorted(mismatches))
        )


def _load_external_subjects(
    manifest_path: Path, lock: Mapping[str, Any]
) -> tuple[tuple[ExternalSubject, ...], str]:
    if _sha256_file(manifest_path) != lock["mipdb_manifest_sha256"]:
        raise ExternalHoldoutError("MIPDB manifest file hash does not match sealed study")
    manifest = _load_json(manifest_path, "MIPDB manifest")
    dataset_manifest_sha256 = manifest.get("dataset_manifest_sha256")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset") != "MIPDB"
        or not _is_sha256(dataset_manifest_sha256)
    ):
        raise ExternalHoldoutError("MIPDB manifest identity is invalid")
    if manifest.get("status") != "finalized":
        raise ExternalHoldoutError("external inference requires a finalized MIPDB manifest")
    if not _is_sha256(manifest.get("cohort_qc_sha256")):
        raise ExternalHoldoutError("finalized MIPDB cohort QC identity is invalid")
    if manifest.get("protocol_sha256") != lock["protocol_sha256"]:
        raise ExternalHoldoutError("MIPDB manifest protocol does not match sealed study")
    cohorts = manifest.get("cohorts")
    subject_hashes = manifest.get("subject_list_sha256")
    if not isinstance(cohorts, Mapping) or not isinstance(subject_hashes, Mapping):
        raise ExternalHoldoutError("MIPDB manifest cohort metadata is incomplete")
    pilot = cohorts.get("pilot")
    primary = cohorts.get("primary")
    extrapolation = cohorts.get("extrapolation")
    if not all(isinstance(value, list) for value in (pilot, primary, extrapolation)):
        raise ExternalHoldoutError("MIPDB manifest cohorts must be ordered arrays")
    cohort_sets = {
        "pilot": set(pilot),
        "primary": set(primary),
        "extrapolation": set(extrapolation),
    }
    if len(pilot) != 10 or len(cohort_sets["pilot"]) != len(pilot):
        raise ExternalHoldoutError("MIPDB pilot must contain exactly 10 unique subjects")
    if len(cohort_sets["primary"]) != len(primary) or not primary:
        raise ExternalHoldoutError("MIPDB primary subject inventory is empty or duplicated")
    if len(cohort_sets["extrapolation"]) != len(extrapolation):
        raise ExternalHoldoutError("MIPDB extrapolation subject inventory is duplicated")
    if any(
        cohort_sets[left] & cohort_sets[right]
        for left, right in (
            ("pilot", "primary"),
            ("pilot", "extrapolation"),
            ("primary", "extrapolation"),
        )
    ):
        raise ExternalHoldoutError(
            "MIPDB pilot/primary/extrapolation cohort contamination detected"
        )
    expected_hashes = {
        "pilot": lock["subject_list_sha256"]["mipdb_pilot"],
        "primary": lock["subject_list_sha256"]["mipdb_primary"],
        "extrapolation": lock["subject_list_sha256"]["mipdb_extrapolation"],
    }
    for name, subjects in (
        ("pilot", pilot),
        ("primary", primary),
        ("extrapolation", extrapolation),
    ):
        claimed = subject_hashes.get(name)
        actual = canonical_sha256(subjects)
        if claimed != actual or actual != expected_hashes[name]:
            raise ExternalHoldoutError(f"MIPDB {name} subject inventory does not match")
    raw_subjects = manifest.get("subjects")
    if not isinstance(raw_subjects, list):
        raise ExternalHoldoutError("MIPDB manifest subjects must be an array")
    by_id: dict[str, float] = {}
    for item in raw_subjects:
        if not isinstance(item, Mapping):
            raise ExternalHoldoutError("MIPDB manifest has an invalid subject record")
        subject_id = item.get("subject_id")
        age = item.get("age")
        if (
            not isinstance(subject_id, str)
            or not _SAFE_SUBJECT_ID.fullmatch(subject_id)
            or subject_id in by_id
            or isinstance(age, bool)
            or not isinstance(age, (int, float))
            or not math.isfinite(float(age))
        ):
            raise ExternalHoldoutError("MIPDB manifest has an invalid subject identity or age")
        by_id[subject_id] = float(age)
    declared_subjects = [*pilot, *primary, *extrapolation]
    missing = [subject_id for subject_id in declared_subjects if subject_id not in by_id]
    if missing:
        raise ExternalHoldoutError(f"MIPDB cohort subjects are missing metadata: {missing}")
    return (
        tuple(ExternalSubject(subject_id, by_id[subject_id]) for subject_id in primary),
        dataset_manifest_sha256,
    )


def _load_heads(
    checkpoint_root: Path,
    inventory_path: Path,
    lock: Mapping[str, Any],
    device: str,
) -> tuple[_LoadedHead, ...]:
    try:
        inventory = load_checkpoint_inventory(inventory_path)
    except StudyLockError as error:
        raise ExternalHoldoutError(str(error)) from error
    if inventory["checkpoint_inventory_sha256"] != lock[
        "checkpoint_inventory_sha256"
    ]:
        raise ExternalHoldoutError("checkpoint inventory does not match sealed study")
    if inventory["training_source_sha256"] != lock["training_source_sha256"]:
        raise ExternalHoldoutError("checkpoint inventory training source does not match")
    if inventory["representation_protocol_sha256"] != lock["protocol_sha256"]:
        raise ExternalHoldoutError(
            "checkpoint inventory representation protocol does not match"
        )
    if inventory["training_protocol_sha256"] != lock["training_protocol_sha256"]:
        raise ExternalHoldoutError(
            "checkpoint inventory training protocol does not match"
        )
    loaded: list[_LoadedHead] = []
    for record in inventory["runs"]:
        head_name = record["head_name"]
        seed = record["seed"]
        checkpoint_path = (
            checkpoint_root / head_name / f"seed-{seed}" / "head_checkpoint.pt"
        )
        if not checkpoint_path.is_file():
            raise ExternalHoldoutError(f"head checkpoint is missing: {checkpoint_path}")
        if _sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
            raise ExternalHoldoutError(f"head checkpoint hash does not match: {checkpoint_path}")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ExternalHoldoutError(f"head checkpoint is unreadable: {checkpoint_path}") from error
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 3
            or not isinstance(payload.get("state_dict"), Mapping)
        ):
            raise ExternalHoldoutError(f"head checkpoint payload is invalid: {checkpoint_path}")
        if (
            payload.get("head_name") != head_name
            or payload.get("seed") != seed
            or payload.get("selected_epoch") != record["selected_epoch"]
            or payload.get("run_identity_sha256") != record["run_identity_sha256"]
            or payload.get("training_source_sha256")
            != lock["training_source_sha256"]
            or payload.get("representation_protocol_sha256")
            != lock["protocol_sha256"]
            or payload.get("training_protocol_sha256")
            != lock["training_protocol_sha256"]
        ):
            raise ExternalHoldoutError(f"head checkpoint identity does not match: {checkpoint_path}")
        linear_weight = payload["state_dict"].get("linear.weight")
        if not isinstance(linear_weight, torch.Tensor) or linear_weight.ndim != 2:
            raise ExternalHoldoutError(f"head checkpoint has no valid linear weight: {checkpoint_path}")
        model = build_frozen_probe_head(
            head_name, embed_dim=int(linear_weight.shape[1])
        )
        try:
            model.load_state_dict(payload["state_dict"], strict=True)
        except (RuntimeError, ValueError) as error:
            raise ExternalHoldoutError(f"head checkpoint state is invalid: {checkpoint_path}") from error
        if sum(parameter.numel() for parameter in model.parameters()) != record[
            "head_parameter_count"
        ]:
            raise ExternalHoldoutError(f"head parameter count does not match: {checkpoint_path}")
        model.to(device)
        model.eval()
        loaded.append(
            _LoadedHead(head_name, seed, record["checkpoint_sha256"], model)
        )
    if {(item.head_name, item.seed) for item in loaded} != {
        (head_name, seed)
        for head_name in APPROVED_HEADS
        for seed in range(33, 43)
    }:
        raise ExternalHoldoutError("loaded head and seed inventory is not exact")
    return tuple(loaded)


def _cache_identity(
    lock: Mapping[str, Any], subject_id: str, dataset_manifest_sha256: str
) -> RepresentationCacheIdentity:
    return RepresentationCacheIdentity(
        protocol_sha256=lock["protocol_sha256"],
        checkpoint=lock["encoder_checkpoint"],
        checkpoint_sha256=lock["encoder_checkpoint_sha256"],
        dataset_manifest_sha256=dataset_manifest_sha256,
        preprocessing_sha256=lock["preprocessing_sha256"],
        subject_id=subject_id,
        source_tree_sha256=lock["training_source_sha256"],
    )


def _prediction_static_fields(
    *,
    lock: Mapping[str, Any],
    subject: ExternalSubject,
    identity: RepresentationCacheIdentity,
    loaded_head: _LoadedHead,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "training_source_sha256": lock["training_source_sha256"],
        "environment_sha256": lock["environment_sha256"],
        "mipdb_manifest_sha256": lock["mipdb_manifest_sha256"],
        "preprocessing_sha256": lock["preprocessing_sha256"],
        "encoder_checkpoint_sha256": lock["encoder_checkpoint_sha256"],
        "head_checkpoint_sha256": loaded_head.checkpoint_sha256,
        "head_name": loaded_head.head_name,
        "seed": loaded_head.seed,
        "subject_id": subject.subject_id,
        "target_age": subject.age,
        "representation_cache_key": identity.key,
    }


def _validate_existing_prediction(
    path: Path, *, expected_static: Mapping[str, Any]
) -> dict[str, Any]:
    record = _load_json(path, "existing prediction")
    expected_fields = {
        *expected_static,
        "qc_status",
        "qc_sha256",
        "prediction",
        "prediction_sha256",
    }
    if set(record) != expected_fields:
        raise ExternalHoldoutError(
            f"existing prediction fields do not match strict schema: {path}"
        )
    claimed = record.get("prediction_sha256")
    body = {key: value for key, value in record.items() if key != "prediction_sha256"}
    if not _is_sha256(claimed) or claimed != canonical_sha256(body):
        raise ExternalHoldoutError(f"existing prediction hash does not match: {path}")
    for key, value in expected_static.items():
        if record.get(key) != value:
            raise ExternalHoldoutError(f"existing prediction identity does not match: {path}")
    if (
        record.get("qc_status") != "passed"
        or not _is_sha256(record.get("qc_sha256"))
        or not isinstance(record.get("prediction"), (int, float))
        or not math.isfinite(float(record["prediction"]))
    ):
        raise ExternalHoldoutError(f"existing prediction content is invalid: {path}")
    return record


def _validate_material(
    material: ExternalSubjectMaterial,
    *,
    subject_id: str,
    expected_identity: RepresentationCacheIdentity,
) -> tuple[dict[int, torch.Tensor], str]:
    if not isinstance(material, ExternalSubjectMaterial):
        raise ExternalHoldoutError("representation provider returned an invalid result")
    if material.cache_identity != expected_identity:
        raise ExternalHoldoutError("external representation cache identity does not match")
    if set(material.representations) != set(PREDECLARED_LAYERS):
        raise ExternalHoldoutError("external representations must contain exactly layers -2 and -1")
    representations: dict[int, torch.Tensor] = {}
    shape: tuple[int, ...] | None = None
    for layer_index in PREDECLARED_LAYERS:
        tensor = material.representations[layer_index]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or tensor.shape[0] == 0
            or tensor.shape[1] == 0
            or tensor.shape[2] == 0
            or not torch.isfinite(tensor).all()
            or tensor.requires_grad
        ):
            raise ExternalHoldoutError("external representation tensor is invalid")
        if shape is not None and tuple(tensor.shape) != shape:
            raise ExternalHoldoutError("external representation layers have different shapes")
        shape = tuple(tensor.shape)
        representations[layer_index] = tensor.detach().cpu().contiguous()
    qc = dict(material.qc)
    if (
        qc.get("status") != "passed"
        or qc.get("subject_id") != subject_id
        or qc.get("window_count") != shape[0]
    ):
        raise ExternalHoldoutError("external subject QC is not a passing exact match")
    return representations, canonical_sha256(qc)


def _predict_subject(
    model: nn.Module, representation: torch.Tensor, *, device: str
) -> float:
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, representation.shape[0], INFERENCE_BATCH_SIZE):
            batch = representation[start : start + INFERENCE_BATCH_SIZE].to(device)
            output = model(batch).reshape(-1)
            if output.numel() != batch.shape[0] or not torch.isfinite(output).all():
                raise ExternalHoldoutError("head produced invalid external predictions")
            values.append(output.detach().cpu())
    prediction = float(torch.cat(values).mean())
    if not math.isfinite(prediction):
        raise ExternalHoldoutError("subject prediction is non-finite")
    return prediction


def _ensure_started(
    lock_path: Path,
    lock: Mapping[str, Any],
    state: Mapping[str, Any],
    output_root: Path,
) -> None:
    if state["state"] == "sealed":
        state = transition_study(lock_path, "started")
    marker_body = {
        "schema_version": 3,
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "state": "started",
        "started_at_utc": state["updated_at_utc"],
    }
    marker = {**marker_body, "marker_sha256": canonical_sha256(marker_body)}
    marker_path = output_root / "evaluation_started.json"
    if marker_path.exists():
        existing = _load_json(marker_path, "evaluation started marker")
        if existing != marker:
            raise ExternalHoldoutError("evaluation started marker does not match study state")
    else:
        _publish_json_create_only(marker_path, marker)


def _complete(
    *,
    lock_path: Path,
    lock: Mapping[str, Any],
    output_root: Path,
    subjects: Sequence[ExternalSubject],
    loaded_heads: Sequence[_LoadedHead],
    dataset_manifest_sha256: str,
) -> Mapping[str, Any]:
    prediction_entries: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for loaded_head in loaded_heads:
        for subject in subjects:
            path = (
                output_root
                / "predictions"
                / loaded_head.head_name
                / f"seed-{loaded_head.seed}"
                / f"{subject.subject_id}.json"
            )
            expected_paths.add(path)
            identity = _cache_identity(
                lock, subject.subject_id, dataset_manifest_sha256
            )
            record = _validate_existing_prediction(
                path,
                expected_static=_prediction_static_fields(
                    lock=lock,
                    subject=subject,
                    identity=identity,
                    loaded_head=loaded_head,
                ),
            )
            prediction_entries.append(
                {
                    "head_name": loaded_head.head_name,
                    "seed": loaded_head.seed,
                    "subject_id": subject.subject_id,
                    "path": str(path.relative_to(output_root)),
                    "prediction_sha256": record["prediction_sha256"],
                }
            )
    actual_paths = set((output_root / "predictions").rglob("*.json"))
    if actual_paths != expected_paths:
        raise ExternalHoldoutError("external prediction file inventory is not exact")
    body: dict[str, Any] = {
        "schema_version": 3,
        "status": "complete",
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "heads": list(APPROVED_HEADS),
        "seeds": list(range(33, 43)),
        "subjects": [subject.subject_id for subject in subjects],
        "prediction_count": len(prediction_entries),
        "predictions": prediction_entries,
    }
    inventory = {**body, "prediction_inventory_sha256": canonical_sha256(body)}
    inventory_path = output_root / "prediction_inventory.json"
    if inventory_path.exists():
        if _load_json(inventory_path, "prediction inventory") != inventory:
            raise ExternalHoldoutError("existing prediction inventory does not match")
    else:
        _publish_json_create_only(inventory_path, inventory)
    completion_body = {
        "schema_version": 3,
        "study_id": lock["study_id"],
        "lock_sha256": lock["lock_sha256"],
        "protocol_sha256": lock["protocol_sha256"],
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "prediction_inventory_sha256": inventory["prediction_inventory_sha256"],
        "status": "complete",
    }
    completion_marker = {
        **completion_body,
        "marker_sha256": canonical_sha256(completion_body),
    }
    completion_path = output_root / "evaluation_completed.json"
    if completion_path.exists():
        if _load_json(completion_path, "evaluation completed marker") != completion_marker:
            raise ExternalHoldoutError(
                "existing evaluation completed marker does not match"
            )
    else:
        _publish_json_create_only(completion_path, completion_marker)
    transition_study(lock_path, "completed")
    return inventory


def load_cached_external_material(
    cache_root: Path,
    subject_id: str,
    identity: RepresentationCacheIdentity,
) -> ExternalSubjectMaterial:
    """Load a complete external cache entry and its subject-level QC evidence."""

    entry = Path(cache_root) / identity.key
    metadata = _load_json(entry / "metadata.json", "representation cache metadata")
    evidence = metadata.get("evidence")
    qc = evidence.get("external_qc") if isinstance(evidence, Mapping) else None
    if not isinstance(qc, Mapping):
        raise ExternalHoldoutError(
            f"representation cache has no external_qc evidence: {entry}"
        )
    try:
        representations = load_cached_representations(cache_root, identity)
    except FrozenEncoderError as error:
        raise ExternalHoldoutError(str(error)) from error
    material = ExternalSubjectMaterial(
        representations=representations,
        cache_identity=identity,
        qc=dict(qc),
    )
    _validate_material(
        material, subject_id=subject_id, expected_identity=identity
    )
    return material


def run_external_holdout(
    *,
    lock_path: Path,
    checkpoint_root: Path,
    inventory_path: Path,
    mipdb_manifest_path: Path,
    environment_path: Path,
    output_root: Path,
    runtime: RuntimeProvenance,
    representation_provider: Callable[
        [str, RepresentationCacheIdentity], ExternalSubjectMaterial
    ],
    device: str,
) -> Mapping[str, Any]:
    """Run or exactly resume sealed external inference without aggregate analysis."""

    try:
        lock = load_study_lock(lock_path)
    except StudyLockError as error:
        raise ExternalHoldoutError(str(error)) from error
    output_root = Path(output_root).resolve()
    if output_root != Path(lock["output_root"]).resolve():
        raise ExternalHoldoutError("external output root does not match sealed study")
    if device not in {"cpu", "cuda", "mps"}:
        raise ExternalHoldoutError("device must be cpu, cuda, or mps")
    if device == "cuda" and not torch.cuda.is_available():
        raise ExternalHoldoutError("CUDA was requested but is unavailable")
    state = _load_state(Path(lock_path), lock)
    _validate_runtime(lock, runtime, Path(environment_path))
    subjects, dataset_manifest_sha256 = _load_external_subjects(
        Path(mipdb_manifest_path), lock
    )
    loaded_heads = _load_heads(
        Path(checkpoint_root), Path(inventory_path), lock, device
    )
    identities = {
        subject.subject_id: _cache_identity(
            lock, subject.subject_id, dataset_manifest_sha256
        )
        for subject in subjects
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_started(Path(lock_path), lock, state, output_root)

    for subject in subjects:
        identity = identities[subject.subject_id]
        existing: dict[tuple[str, int], dict[str, Any]] = {}
        missing: list[_LoadedHead] = []
        for loaded_head in loaded_heads:
            path = (
                output_root
                / "predictions"
                / loaded_head.head_name
                / f"seed-{loaded_head.seed}"
                / f"{subject.subject_id}.json"
            )
            static = _prediction_static_fields(
                lock=lock,
                subject=subject,
                identity=identity,
                loaded_head=loaded_head,
            )
            if path.exists():
                existing[(loaded_head.head_name, loaded_head.seed)] = (
                    _validate_existing_prediction(path, expected_static=static)
                )
            else:
                missing.append(loaded_head)
        if not missing:
            continue
        material = representation_provider(subject.subject_id, identity)
        representations, qc_sha256 = _validate_material(
            material,
            subject_id=subject.subject_id,
            expected_identity=identity,
        )
        if any(record["qc_sha256"] != qc_sha256 for record in existing.values()):
            raise ExternalHoldoutError("resumed subject QC does not match existing predictions")
        for loaded_head in missing:
            layer_index = required_layer_for_head(loaded_head.head_name)
            prediction = _predict_subject(
                loaded_head.model,
                representations[layer_index],
                device=device,
            )
            static = _prediction_static_fields(
                lock=lock,
                subject=subject,
                identity=identity,
                loaded_head=loaded_head,
            )
            body = {
                **static,
                "qc_status": "passed",
                "qc_sha256": qc_sha256,
                "prediction": prediction,
            }
            path = (
                output_root
                / "predictions"
                / loaded_head.head_name
                / f"seed-{loaded_head.seed}"
                / f"{subject.subject_id}.json"
            )
            _publish_json_create_only(
                path,
                {**body, "prediction_sha256": canonical_sha256(body)},
            )
    return _complete(
        lock_path=Path(lock_path),
        lock=lock,
        output_root=output_root,
        subjects=subjects,
        loaded_heads=loaded_heads,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
