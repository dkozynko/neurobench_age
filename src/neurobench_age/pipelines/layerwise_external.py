"""Lock-gated external evaluation for the secondary layer-wise study."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from neurobench_age.pipelines.external_holdout import _validate_material
from neurobench_age.pipelines.frozen_probe import (
    RepresentationCacheIdentity,
    _is_sha256,
    load_cached_representations,
)
from neurobench_age.pipelines.representation_materialization import (
    preprocessing_contract_sha256,
)
from neurobench_age.pipelines.representation_materialization import (
    LazyMipdbRepresentationProvider,
)
from neurobench_age.research.layerwise_probe import LayerwiseHeadSpec
from neurobench_age.research.protocol import StudyProtocol
from neurobench_age.research.training_protocol import FrozenProbeTrainingProtocol


class LayerwiseExternalError(RuntimeError):
    """Raised when the external layer-wise evidence is missing or inconsistent."""


_SAFE_SUBJECT_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_INFERENCE_BATCH_SIZE = 64


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LayerwiseExternalError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise LayerwiseExternalError(f"{description} must be an object")
    return value


def _validated_subject_inventory(manifest: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    raw_subjects = manifest.get("subjects")
    if not isinstance(raw_subjects, list):
        raise LayerwiseExternalError("HBN subject inventory is invalid")
    inventory: list[tuple[Any, ...]] = []
    for item in raw_subjects:
        if not isinstance(item, Mapping) or set(item) != {"subject_id", "split", "age"}:
            raise LayerwiseExternalError("HBN subject inventory contains an invalid record")
        subject_id = item.get("subject_id")
        split = item.get("split")
        if not isinstance(subject_id, str) or not subject_id:
            raise LayerwiseExternalError("HBN subject inventory contains an invalid subject")
        if not isinstance(split, str) or not split:
            raise LayerwiseExternalError("HBN subject inventory contains an invalid split")
        inventory.append((subject_id, split, _finite_float(item.get("age"), "HBN age")))
    if len({item[0] for item in inventory}) != len(inventory):
        raise LayerwiseExternalError("HBN subject inventory contains duplicate subjects")
    return tuple(sorted(inventory))


def _validated_acquisition_inventory(
    manifest: Mapping[str, Any], *, subject_manifest_sha256: str
) -> tuple[tuple[Any, ...], ...]:
    raw_files = manifest.get("acquisition_files")
    if not isinstance(raw_files, list) or not raw_files:
        raise LayerwiseExternalError("HBN acquisition inventory is invalid")
    inventory: list[tuple[Any, ...]] = []
    for item in raw_files:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"subject_id", "path", "size_bytes", "sha256"}
            or not isinstance(item.get("subject_id"), str)
            or not isinstance(item.get("path"), str)
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item.get("size_bytes") < 0
            or not isinstance(item.get("sha256"), str)
            or not _is_sha256(item.get("sha256"))
        ):
            raise LayerwiseExternalError("HBN acquisition inventory contains an invalid record")
        inventory.append(
            (
                item["subject_id"],
                item["path"],
                item["size_bytes"],
                item["sha256"],
            )
        )
    if len(set(inventory)) != len(inventory):
        raise LayerwiseExternalError("HBN acquisition inventory contains duplicates")
    expected_dataset_sha256 = _canonical_sha256(
        {
            "subject_manifest_sha256": subject_manifest_sha256,
            "acquisition_files": [
                {
                    "subject_id": subject_id,
                    "path": path,
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
                for subject_id, path, size_bytes, sha256 in inventory
            ],
        }
    )
    if manifest.get("dataset_manifest_sha256") != expected_dataset_sha256:
        raise LayerwiseExternalError("HBN acquisition inventory digest is invalid")
    return tuple(sorted(inventory))


def _validate_hbn_training_manifest_alignment(
    layerwise_manifest: Mapping[str, Any],
    primary_manifest: Mapping[str, Any],
) -> str:
    """Require equal HBN subjects and files, independent of manifest ordering."""

    layerwise_subject_sha256 = layerwise_manifest.get("subject_manifest_sha256")
    primary_subject_sha256 = primary_manifest.get("subject_manifest_sha256")
    if (
        not isinstance(layerwise_subject_sha256, str)
        or not _is_sha256(layerwise_subject_sha256)
        or primary_subject_sha256 != layerwise_subject_sha256
    ):
        raise LayerwiseExternalError("HBN subject manifests do not match")
    if _validated_subject_inventory(layerwise_manifest) != _validated_subject_inventory(
        primary_manifest
    ):
        raise LayerwiseExternalError("HBN subject inventories do not match")
    layerwise_files = _validated_acquisition_inventory(
        layerwise_manifest, subject_manifest_sha256=layerwise_subject_sha256
    )
    primary_files = _validated_acquisition_inventory(
        primary_manifest, subject_manifest_sha256=primary_subject_sha256
    )
    if layerwise_files != primary_files:
        raise LayerwiseExternalError("HBN acquisition inventory does not match")
    if layerwise_manifest.get("dataset_manifest_sha256") == primary_manifest.get(
        "dataset_manifest_sha256"
    ):
        return "exact_dataset_identity"
    return "normalized_acquisition_inventory"


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise LayerwiseExternalError(f"immutable artifact already exists: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayerwiseExternalError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise LayerwiseExternalError(f"{field} must be finite")
    return value


def _load_primary_subjects(
    manifest_path: Path, *, manifest_protocol_sha256: str
) -> tuple[tuple[dict[str, Any], ...], str]:
    manifest = _load_json(manifest_path, "MIPDB manifest")
    dataset_sha256 = manifest.get("dataset_manifest_sha256")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("dataset") != "MIPDB"
        or manifest.get("status") != "finalized"
        or not isinstance(dataset_sha256, str)
        or not _is_sha256(dataset_sha256)
        or manifest.get("protocol_sha256") != manifest_protocol_sha256
    ):
        raise LayerwiseExternalError("MIPDB manifest identity is invalid")
    cohorts = manifest.get("cohorts")
    raw_subjects = manifest.get("subjects")
    primary = cohorts.get("primary") if isinstance(cohorts, Mapping) else None
    if not isinstance(primary, list) or len(primary) != 75 or len(set(primary)) != 75:
        raise LayerwiseExternalError("layer-wise external evaluation requires 75 primary MIPDB subjects")
    if not isinstance(raw_subjects, list):
        raise LayerwiseExternalError("MIPDB subject inventory is invalid")
    by_id: dict[str, float] = {}
    for item in raw_subjects:
        if not isinstance(item, Mapping):
            raise LayerwiseExternalError("MIPDB subject record is invalid")
        subject_id = item.get("subject_id")
        if (
            not isinstance(subject_id, str)
            or not _SAFE_SUBJECT_ID.fullmatch(subject_id)
            or subject_id in by_id
        ):
            raise LayerwiseExternalError("MIPDB subject identity is invalid")
        by_id[subject_id] = _finite_float(item.get("age"), "MIPDB age")
    if any(subject_id not in by_id for subject_id in primary):
        raise LayerwiseExternalError("MIPDB primary cohort has missing age metadata")
    return (
        tuple(
            {"subject_id": subject_id, "age": by_id[subject_id], "split": "mipdb_primary"}
            for subject_id in primary
        ),
        dataset_sha256,
    )


def _discover_primary_cache_source_tree_sha256(
    cache_root: Path,
    *,
    protocol_sha256: str,
    checkpoint: str,
    checkpoint_sha256: str,
    dataset_manifest_sha256: str,
    preprocessing_sha256: str,
) -> str:
    """Read the immutable source identity recorded by the sealed MIPDB cache.

    The primary MIPDB representations may have been materialized before the
    current checkout was created.  Their source identity is therefore not
    required to equal the current HBN training manifest's source identity; it
    must instead be validated from the cache entries that are actually used.
    """

    cache_root = Path(cache_root)
    entries = sorted(path for path in cache_root.iterdir() if path.is_dir())
    if not entries:
        raise LayerwiseExternalError("primary MIPDB cache has no entries")
    observed_source_tree_sha256: str | None = None
    for entry in entries:
        metadata = _load_json(entry / "metadata.json", "primary MIPDB cache metadata")
        identity = metadata.get("identity")
        if (
            metadata.get("schema_version") != 1
            or metadata.get("status") != "complete"
            or not isinstance(identity, Mapping)
            or metadata.get("cache_key") != entry.name
            or identity.get("protocol_sha256") != protocol_sha256
            or identity.get("checkpoint") != checkpoint
            or identity.get("checkpoint_sha256") != checkpoint_sha256
            or identity.get("dataset_manifest_sha256") != dataset_manifest_sha256
            or identity.get("preprocessing_sha256") != preprocessing_sha256
            or not isinstance(identity.get("source_tree_sha256"), str)
            or not _is_sha256(identity["source_tree_sha256"])
        ):
            raise LayerwiseExternalError(
                f"primary MIPDB cache identity is invalid: {entry}"
            )
        source_tree_sha256 = identity["source_tree_sha256"]
        if (
            observed_source_tree_sha256 is not None
            and source_tree_sha256 != observed_source_tree_sha256
        ):
            raise LayerwiseExternalError(
                "primary MIPDB cache entries have inconsistent source identities"
            )
        observed_source_tree_sha256 = source_tree_sha256
    assert observed_source_tree_sha256 is not None
    return observed_source_tree_sha256


def _load_checkpoint(
    checkpoint_root: Path,
    run: Mapping[str, Any],
    *,
    protocol: StudyProtocol,
    training: FrozenProbeTrainingProtocol,
    training_source_sha256: str,
    device: str,
) -> nn.Module:
    path = Path(checkpoint_root) / str(run["head_name"]) / f"seed-{run['seed']}" / "head_checkpoint.pt"
    if not path.is_file() or _sha256_file(path) != run["checkpoint_sha256"]:
        raise LayerwiseExternalError(f"layer-wise checkpoint is missing or changed: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise LayerwiseExternalError(f"layer-wise checkpoint is unreadable: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 3:
        raise LayerwiseExternalError(f"layer-wise checkpoint payload is invalid: {path}")
    expected = {
        "head_name": run["head_name"],
        "seed": run["seed"],
        "representation_protocol_sha256": protocol.sha256,
        "training_protocol_sha256": training.sha256,
        "training_source_sha256": training_source_sha256,
        "run_identity_sha256": run["run_identity_sha256"],
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise LayerwiseExternalError(f"layer-wise checkpoint provenance differs: {path}")
    state_dict = payload.get("state_dict")
    weight = state_dict.get("linear.weight") if isinstance(state_dict, Mapping) else None
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise LayerwiseExternalError(f"layer-wise checkpoint has no linear weight: {path}")
    from neurobench_age.heads.math import MeanLinearCopyHead

    model = MeanLinearCopyHead(embed_dim=int(weight.shape[1]), n_outputs=int(weight.shape[0]))
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as error:
        raise LayerwiseExternalError(f"layer-wise checkpoint state is invalid: {path}") from error
    model.to(device).eval()
    return model


def _predict(model: nn.Module, representation: torch.Tensor, *, device: str) -> float:
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, representation.shape[0], _INFERENCE_BATCH_SIZE):
            batch = representation[start : start + _INFERENCE_BATCH_SIZE].to(device)
            output = model(batch).reshape(-1)
            if output.numel() != batch.shape[0] or not torch.isfinite(output).all():
                raise LayerwiseExternalError("layer-wise head emitted invalid predictions")
            values.append(output.detach().cpu())
    prediction = float(torch.cat(values).mean())
    if not math.isfinite(prediction):
        raise LayerwiseExternalError("layer-wise prediction is non-finite")
    return prediction


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target = torch.tensor([float(row["true_age"]) for row in rows], dtype=torch.float64)
    prediction = torch.tensor([float(row["prediction"]) for row in rows], dtype=torch.float64)
    centered_target = target - target.mean()
    centered_prediction = prediction - prediction.mean()
    denominator = float(torch.sqrt(centered_target.square().sum() * centered_prediction.square().sum()))
    if denominator <= 0.0:
        raise LayerwiseExternalError("external Pearson is undefined")
    return {
        "subject_count": len(rows),
        "pearson": float(torch.dot(centered_target, centered_prediction) / denominator),
        "mae": float((prediction - target).abs().mean()),
        "rmse": float(torch.sqrt((prediction - target).square().mean())),
    }


def run_layerwise_external(
    *,
    protocol: StudyProtocol,
    training: FrozenProbeTrainingProtocol,
    checkpoint_root: Path,
    checkpoint_inventory_path: Path,
    mipdb_manifest_path: Path,
    manifest_protocol_sha256: str,
    training_manifest_path: Path,
    primary_protocol: StudyProtocol,
    primary_training_manifest_path: Path,
    primary_cache_root: Path,
    bids_root: Path,
    mapping_path: Path,
    cache_root: Path,
    output_root: Path,
    training_source_sha256: str,
    evaluation_source_sha256: str,
    device: str = "cpu",
) -> Mapping[str, Any]:
    """Evaluate all layer-by-seed checkpoints on the sealed 75-subject primary cohort."""

    if device not in {"cpu", "cuda", "mps"}:
        raise LayerwiseExternalError("device must be cpu, cuda, or mps")
    if device == "cuda" and not torch.cuda.is_available():
        raise LayerwiseExternalError("CUDA was requested but is unavailable")
    if not _is_sha256(training_source_sha256):
        raise LayerwiseExternalError("training source identity is invalid")
    if not _is_sha256(evaluation_source_sha256):
        raise LayerwiseExternalError("evaluation source identity is invalid")
    inventory = _load_json(checkpoint_inventory_path, "layer-wise checkpoint inventory")
    required_fields = {
        "schema_version", "status", "study_id", "protocol_sha256",
        "training_protocol_sha256", "training_source_sha256", "layer_indices",
        "heads", "head_layers", "seeds", "run_count", "runs",
        "checkpoint_inventory_sha256",
    }
    body = {key: value for key, value in inventory.items() if key != "checkpoint_inventory_sha256"}
    if set(inventory) != required_fields or inventory.get("checkpoint_inventory_sha256") != _canonical_sha256(body):
        raise LayerwiseExternalError("layer-wise checkpoint inventory digest/schema is invalid")
    if (
        inventory["status"] != "complete"
        or inventory["study_id"] != protocol.study_id
        or inventory["protocol_sha256"] != protocol.sha256
        or inventory["training_protocol_sha256"] != training.sha256
        or inventory["training_source_sha256"] != training_source_sha256
        or tuple(inventory["layer_indices"]) != tuple(protocol.encoder.layer_indices)
        or tuple(inventory["seeds"]) != tuple(training.seeds)
        or inventory["run_count"] != len(protocol.heads) * len(training.seeds)
    ):
        raise LayerwiseExternalError("layer-wise checkpoint inventory does not match inputs")
    runs = tuple(inventory["runs"])
    if len(runs) != inventory["run_count"]:
        raise LayerwiseExternalError("layer-wise checkpoint run inventory is incomplete")

    subjects, dataset_sha256 = _load_primary_subjects(
        mipdb_manifest_path, manifest_protocol_sha256=manifest_protocol_sha256
    )
    training_manifest = _load_json(training_manifest_path, "HBN training manifest")
    encoder_checkpoint_sha256 = training_manifest.get("checkpoint_sha256")
    representation_source_tree_sha256 = training_manifest.get("source_tree_sha256")
    if (
        training_manifest.get("protocol_sha256") != protocol.sha256
        or training_manifest.get("checkpoint") != protocol.encoder.checkpoint
        or not isinstance(encoder_checkpoint_sha256, str)
        or not _is_sha256(encoder_checkpoint_sha256)
        or not isinstance(representation_source_tree_sha256, str)
        or not _is_sha256(representation_source_tree_sha256)
        or training_manifest.get("preprocessing_sha256")
        != preprocessing_contract_sha256(protocol.preprocessing)
    ):
        raise LayerwiseExternalError("HBN training manifest does not match layer-wise inputs")
    primary_training_manifest = _load_json(
        primary_training_manifest_path, "primary HBN training manifest"
    )
    primary_checkpoint_sha256 = primary_training_manifest.get("checkpoint_sha256")
    primary_source_tree_sha256 = primary_training_manifest.get("source_tree_sha256")
    if (
        primary_training_manifest.get("protocol_sha256") != primary_protocol.sha256
        or primary_training_manifest.get("checkpoint") != primary_protocol.encoder.checkpoint
        or not isinstance(primary_checkpoint_sha256, str)
        or not _is_sha256(primary_checkpoint_sha256)
        or not isinstance(primary_source_tree_sha256, str)
        or not _is_sha256(primary_source_tree_sha256)
    ):
        raise LayerwiseExternalError("primary HBN training manifest does not match external inputs")
    hbn_dataset_alignment = _validate_hbn_training_manifest_alignment(
        training_manifest, primary_training_manifest
    )
    primary_cache_source_tree_sha256 = _discover_primary_cache_source_tree_sha256(
        primary_cache_root,
        protocol_sha256=primary_protocol.sha256,
        checkpoint=primary_protocol.encoder.checkpoint,
        checkpoint_sha256=primary_checkpoint_sha256,
        dataset_manifest_sha256=dataset_sha256,
        preprocessing_sha256=preprocessing_contract_sha256(
            primary_protocol.preprocessing
        ),
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_body = {
        "schema_version": 1,
        "status": "sealed",
        "study_id": protocol.study_id,
        "protocol_sha256": protocol.sha256,
        "training_protocol_sha256": training.sha256,
        "training_source_sha256": training_source_sha256,
        "evaluation_source_tree_sha256": evaluation_source_sha256,
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "preprocessing_sha256": preprocessing_contract_sha256(protocol.preprocessing),
        "hbn_training_manifest_sha256": _sha256_file(training_manifest_path),
        "hbn_training_dataset_manifest_sha256": training_manifest[
            "dataset_manifest_sha256"
        ],
        "hbn_representation_source_tree_sha256": representation_source_tree_sha256,
        "primary_hbn_training_manifest_sha256": _sha256_file(primary_training_manifest_path),
        "primary_hbn_training_dataset_manifest_sha256": primary_training_manifest[
            "dataset_manifest_sha256"
        ],
        "hbn_dataset_alignment": hbn_dataset_alignment,
        "primary_hbn_training_source_tree_sha256": primary_source_tree_sha256,
        "primary_cache_source_tree_sha256": primary_cache_source_tree_sha256,
        "primary_protocol_sha256": primary_protocol.sha256,
        "checkpoint_inventory_sha256": inventory["checkpoint_inventory_sha256"],
        "mipdb_manifest_sha256": _sha256_file(mipdb_manifest_path),
        "mipdb_dataset_manifest_sha256": dataset_sha256,
        "mipdb_primary_subject_list_sha256": _canonical_sha256(
            [subject["subject_id"] for subject in subjects]
        ),
        "layer_indices": list(protocol.encoder.layer_indices),
        "head_layers": dict(inventory["head_layers"]),
        "seeds": list(training.seeds),
        "primary_subject_count": len(subjects),
        "output_root": str(output_root.resolve()),
    }
    lock = {**lock_body, "lock_sha256": _canonical_sha256(lock_body)}
    lock_path = output_root / "layerwise_external_lock.json"
    if lock_path.exists():
        if _load_json(lock_path, "layer-wise external lock") != lock:
            raise LayerwiseExternalError("existing layer-wise external lock conflicts")
    else:
        _write_create_only(lock_path, lock)
    marker_path = output_root / "evaluation_started.json"
    marker = {"schema_version": 1, "state": "started", "lock_sha256": lock["lock_sha256"]}
    if marker_path.exists():
        if _load_json(marker_path, "layer-wise evaluation marker") != marker:
            raise LayerwiseExternalError("existing layer-wise evaluation marker conflicts")
    else:
        _write_create_only(marker_path, marker)

    specs = tuple(
        LayerwiseHeadSpec(head.name, head.layer_index, head.aggregation)
        for head in protocol.heads
    )
    layer_by_head = {spec.name: spec.layer_index for spec in specs}
    provider = LazyMipdbRepresentationProvider(
        protocol=protocol,
        bids_root=bids_root,
        manifest_path=mipdb_manifest_path,
        cache_root=cache_root,
        mapping_path=mapping_path,
        started_marker_path=marker_path,
        expected_lock_sha256=lock["lock_sha256"],
        device=device,
        extraction_batch_size=8,
        expected_manifest_protocol_sha256=manifest_protocol_sha256,
        materialized_layers=(-4, -3),
        pool_tokens=True,
    )
    loaded_models = {
        (run["head_name"], run["seed"]): _load_checkpoint(
            checkpoint_root,
            run,
            protocol=protocol,
            training=training,
            training_source_sha256=training_source_sha256,
            device=device,
        )
        for run in runs
    }
    all_rows: list[dict[str, Any]] = []
    for subject in subjects:
        missing = []
        existing_rows: dict[tuple[str, int], dict[str, Any]] = {}
        for run in runs:
            path = output_root / "predictions" / str(run["head_name"]) / f"seed-{run['seed']}" / f"{subject['subject_id']}.json"
            if path.exists():
                record = _load_json(path, "existing layer-wise prediction")
                body = {key: value for key, value in record.items() if key != "body_sha256"}
                if record.get("body_sha256") != _canonical_sha256(body):
                    raise LayerwiseExternalError(f"prediction digest is invalid: {path}")
                expected_static = {
                    "schema_version": 1,
                    "study_id": protocol.study_id,
                    "lock_sha256": lock["lock_sha256"],
                    "protocol_sha256": protocol.sha256,
                    "training_protocol_sha256": training.sha256,
                    "training_source_sha256": training_source_sha256,
                    "head_name": run["head_name"],
                    "layer_index": run["layer_index"],
                    "seed": run["seed"],
                    "subject_id": subject["subject_id"],
                    "true_age": subject["age"],
                }
                if any(record.get(key) != value for key, value in expected_static.items()):
                    raise LayerwiseExternalError(f"prediction identity is invalid: {path}")
                if not _is_sha256(record.get("body_sha256")) or not math.isfinite(float(record.get("prediction", float("nan")))):
                    raise LayerwiseExternalError(f"prediction content is invalid: {path}")
                existing_rows[(run["head_name"], run["seed"])] = record
            else:
                missing.append(run)
        if missing:
            identity = RepresentationCacheIdentity(
                protocol_sha256=protocol.sha256,
                checkpoint=protocol.encoder.checkpoint,
                checkpoint_sha256=encoder_checkpoint_sha256,
                dataset_manifest_sha256=dataset_sha256,
                preprocessing_sha256=preprocessing_contract_sha256(protocol.preprocessing),
                subject_id=subject["subject_id"],
                source_tree_sha256=training_source_sha256,
            )
            material = provider(subject["subject_id"], identity)
            representations, _qc_sha256 = _validate_material(
                material,
                subject_id=subject["subject_id"],
                expected_identity=identity,
                required_layers=(-4, -3),
            )
            primary_identity = RepresentationCacheIdentity(
                protocol_sha256=primary_protocol.sha256,
                checkpoint=primary_protocol.encoder.checkpoint,
                checkpoint_sha256=primary_checkpoint_sha256,
                dataset_manifest_sha256=dataset_sha256,
                preprocessing_sha256=preprocessing_contract_sha256(
                    primary_protocol.preprocessing
                ),
                subject_id=subject["subject_id"],
                source_tree_sha256=primary_cache_source_tree_sha256,
            )
            primary_representations = load_cached_representations(
                primary_cache_root,
                primary_identity,
                required_layers=(-2, -1),
            )
            representations = {
                -4: representations[-4],
                -3: representations[-3],
                -2: primary_representations[-2].mean(dim=1, keepdim=True).contiguous(),
                -1: primary_representations[-1].mean(dim=1, keepdim=True).contiguous(),
            }
            shape = tuple(representations[-4].shape)
            if any(tuple(tensor.shape) != shape for tensor in representations.values()):
                raise LayerwiseExternalError(
                    f"pooled layer shapes differ for subject {subject['subject_id']}"
                )
            for run in missing:
                body = {
                    "schema_version": 1,
                    "study_id": protocol.study_id,
                    "lock_sha256": lock["lock_sha256"],
                    "protocol_sha256": protocol.sha256,
                    "training_protocol_sha256": training.sha256,
                    "training_source_sha256": training_source_sha256,
                    "head_name": run["head_name"],
                    "layer_index": run["layer_index"],
                    "seed": run["seed"],
                    "subject_id": subject["subject_id"],
                    "true_age": subject["age"],
                    "prediction": _predict(
                        loaded_models[(run["head_name"], run["seed"])],
                        representations[run["layer_index"]],
                        device=device,
                    ),
                }
                path = (
                    output_root
                    / "predictions"
                    / str(run["head_name"])
                    / f"seed-{run['seed']}"
                    / f"{subject['subject_id']}.json"
                )
                _write_create_only(
                    path, {**body, "body_sha256": _canonical_sha256(body)}
                )
                existing_rows[(run["head_name"], run["seed"])] = {
                    **body,
                    "body_sha256": _canonical_sha256(body),
                }
        all_rows.extend(existing_rows.values())

    metrics: list[dict[str, Any]] = []
    for run in runs:
        rows = [
            row for row in all_rows
            if row["head_name"] == run["head_name"] and row["seed"] == run["seed"]
        ]
        metrics.append({
            "head_name": run["head_name"],
            "layer_index": layer_by_head[run["head_name"]],
            "seed": run["seed"],
            **_metrics(rows),
        })
    metrics_body = {
        "schema_version": 1,
        "status": "complete",
        "lock_sha256": lock["lock_sha256"],
        "runs": metrics,
    }
    metrics_payload = {**metrics_body, "metrics_sha256": _canonical_sha256(metrics_body)}
    prediction_body = {
        "schema_version": 1,
        "status": "complete",
        "lock_sha256": lock["lock_sha256"],
        "prediction_count": len(all_rows),
        "prediction_file_count": len(runs) * len(subjects),
        "prediction_files_sha256": _canonical_sha256(
            sorted(
                _canonical_sha256(row)
                for row in all_rows
            )
        ),
    }
    prediction_inventory = {
        **prediction_body,
        "prediction_inventory_sha256": _canonical_sha256(prediction_body),
    }
    prediction_inventory_path = output_root / "prediction_inventory.json"
    if prediction_inventory_path.exists():
        if _load_json(prediction_inventory_path, "layer-wise prediction inventory") != prediction_inventory:
            raise LayerwiseExternalError("existing layer-wise prediction inventory conflicts")
    else:
        _write_create_only(prediction_inventory_path, prediction_inventory)
    metrics_path = output_root / "external_metrics.json"
    if metrics_path.exists():
        if _load_json(metrics_path, "layer-wise external metrics") != metrics_payload:
            raise LayerwiseExternalError("existing layer-wise external metrics conflict")
    else:
        _write_create_only(metrics_path, metrics_payload)
    return {"status": "complete", "lock": lock, "metrics": metrics_payload}
