"""Auditable HBN/MIPDB representation materialization for frozen REVE."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.data.medium_subset import manifest_sha256, read_manifest
from neurobench_age.data.mipdb import (
    MipdbInventoryError,
    load_mipdb_resting_subject,
    verify_mipdb_acquisition,
)
from neurobench_age.pipelines.independent import (
    HbnRecording,
    PreparedRecording,
    PreprocessedRecordingStore,
    hbn_recording_source_files,
)
from neurobench_age.research.protocol import PreprocessingContract, StudyProtocol

from .external_holdout import ExternalSubjectMaterial, load_cached_external_material
from .frozen_probe import (
    PREDECLARED_LAYERS,
    FrozenEncoderError,
    RepresentationCacheIdentity,
    _canonical_sha256,
    _is_sha256,
    assert_frozen_encoder,
    encoder_state_sha256,
    extract_frozen_representations,
    load_cached_representations,
    load_reve_encoder,
    write_cached_representations,
)


def preprocessing_contract_sha256(contract: PreprocessingContract) -> str:
    """Return the canonical identity used by every representation cache."""

    return _canonical_sha256(asdict(contract))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_frozen_representations_batched(
    encoder: nn.Module,
    windows: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    device: str,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Extract all declared layers with bounded accelerator memory."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise FrozenEncoderError("extraction batch_size must be a positive integer")
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[0] == 0 or not torch.isfinite(tensor).all():
        raise FrozenEncoderError(
            "extraction windows must be a finite [windows, channels, samples] tensor"
        )
    encoder.to(device)
    state_before = encoder_state_sha256(encoder)
    chunks: dict[int, list[torch.Tensor]] = {index: [] for index in PREDECLARED_LAYERS}
    for start in range(0, tensor.shape[0], batch_size):
        representations, evidence = extract_frozen_representations(
            encoder,
            tensor[start : start + batch_size].to(device),
        )
        if (
            evidence["state_sha256_before"] != state_before
            or evidence["state_sha256_after"] != state_before
        ):
            raise FrozenEncoderError("encoder state changed between extraction batches")
        for index in PREDECLARED_LAYERS:
            chunks[index].append(representations[index])
    state_after = encoder_state_sha256(encoder)
    assert_frozen_encoder(encoder, expected_state_sha256=state_before)
    return (
        {index: torch.cat(chunks[index], dim=0) for index in PREDECLARED_LAYERS},
        {
            "encoder_frozen": True,
            "encoder_eval_mode": True,
            "inference_mode": True,
            "layer_indices": list(PREDECLARED_LAYERS),
            "state_sha256_before": state_before,
            "state_sha256_after": state_after,
            "extraction_batch_size": batch_size,
        },
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenEncoderError(f"could not read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise FrozenEncoderError(f"{description} must contain a JSON object")
    return payload


def _contains_forbidden_pilot_field(value: object) -> bool:
    forbidden = {"age", "target", "targets", "prediction", "predictions", "metrics"}
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in forbidden or _contains_forbidden_pilot_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_pilot_field(item) for item in value)
    return False


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a JSON artifact without replacing prior evidence."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FrozenEncoderError(f"output already exists: {path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_mipdb_pilot(
    *,
    protocol: StudyProtocol,
    bids_root: Path,
    manifest_path: Path,
    mapping_path: Path,
    output_path: Path,
    device: str,
    extraction_batch_size: int,
    subject_loader: Callable[..., tuple[np.ndarray, dict[str, Any]]] = load_mipdb_resting_subject,
    encoder_loader: Callable[..., nn.Module] = load_reve_encoder,
) -> dict[str, Any]:
    """Run signal/model QC on exactly the target-free, predeclared MIPDB pilot."""

    if device not in {"cpu", "cuda", "mps"}:
        raise FrozenEncoderError("pilot extraction device must be cpu, cuda, or mps")
    if (
        isinstance(extraction_batch_size, bool)
        or not isinstance(extraction_batch_size, int)
        or extraction_batch_size <= 0
    ):
        raise FrozenEncoderError("pilot extraction batch_size must be positive")
    if Path(output_path).exists():
        raise FrozenEncoderError(f"output already exists: {output_path}")

    manifest = _load_json(manifest_path, "MIPDB manifest")
    if manifest.get("dataset") != "MIPDB":
        raise FrozenEncoderError("pilot requires a MIPDB manifest")
    if manifest.get("schema_version") != 2 or manifest.get("status") != "draft":
        raise FrozenEncoderError("pilot requires the exact draft MIPDB manifest")
    if manifest.get("protocol_sha256") != protocol.sha256:
        raise FrozenEncoderError("MIPDB manifest protocol does not match")
    dataset_sha256 = manifest.get("dataset_manifest_sha256")
    if not isinstance(dataset_sha256, str) or not _is_sha256(dataset_sha256):
        raise FrozenEncoderError("MIPDB manifest dataset identity is invalid")
    if subject_loader is load_mipdb_resting_subject:
        try:
            verify_mipdb_acquisition(Path(bids_root), manifest)
        except MipdbInventoryError as error:
            raise FrozenEncoderError(str(error)) from error

    cohorts = manifest.get("cohorts")
    subjects_raw = manifest.get("subjects")
    if not isinstance(cohorts, Mapping) or not isinstance(subjects_raw, list):
        raise FrozenEncoderError("MIPDB manifest cohorts or subjects are invalid")
    pilot = cohorts.get("pilot")
    primary = cohorts.get("primary")
    extrapolation = cohorts.get("extrapolation")
    if not all(isinstance(value, list) for value in (pilot, primary, extrapolation)):
        raise FrozenEncoderError("MIPDB cohorts must be ordered arrays")
    assert isinstance(pilot, list)
    assert isinstance(primary, list)
    assert isinstance(extrapolation, list)
    if (
        len(pilot) != protocol.datasets.pilot_size
        or not all(isinstance(item, str) and item.strip() for item in pilot)
        or len(set(pilot)) != len(pilot)
    ):
        raise FrozenEncoderError(
            f"MIPDB pilot must contain exactly {protocol.datasets.pilot_size} unique subjects"
        )
    if set(pilot) & (set(primary) | set(extrapolation)):
        raise FrozenEncoderError("MIPDB pilot overlaps an evaluation cohort")

    subjects: dict[str, Mapping[str, Any]] = {}
    for item in subjects_raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("subject_id"), str):
            raise FrozenEncoderError("MIPDB manifest has an invalid subject record")
        subject_id = item["subject_id"]
        if subject_id in subjects:
            raise FrozenEncoderError("MIPDB manifest has duplicate subjects")
        subjects[subject_id] = item
    if any(subject_id not in subjects for subject_id in pilot):
        raise FrozenEncoderError("MIPDB pilot subject is absent from the manifest")

    encoder: nn.Module | None = None
    channel_labels: tuple[str, ...] | None = None
    encoder_sha256: str | None = None
    subject_reports: list[dict[str, Any]] = []
    for subject_id in pilot:
        windows, raw_qc = subject_loader(
            Path(bids_root),
            subjects[subject_id],
            contract=protocol.preprocessing,
        )
        windows_array = np.asarray(windows)
        if _contains_forbidden_pilot_field(raw_qc):
            raise FrozenEncoderError(
                "pilot QC contains a target, prediction, or metric field"
            )
        if not isinstance(raw_qc, Mapping):
            raise FrozenEncoderError("pilot QC must be a mapping")
        qc = dict(raw_qc)
        if qc.get("window_count") != int(windows_array.shape[0]):
            raise FrozenEncoderError("pilot QC window count does not match material")
        if "status" in qc and qc["status"] != "passed":
            raise FrozenEncoderError("pilot QC status is not passed")
        if qc.get("qc_reasons") != []:
            raise FrozenEncoderError("pilot QC contains rejection reasons")
        if (
            qc.get("mapped_channel_count") != 128
            or qc.get("cross_block_windows") is not False
            or qc.get("spatial_interpolation") is not False
        ):
            raise FrozenEncoderError("pilot QC violates the preprocessing contract")
        labels = qc.get("channel_labels")
        if (
            not isinstance(labels, list)
            or labels != [f"E{index}" for index in range(1, 129)]
        ):
            raise FrozenEncoderError("pilot requires the exact E1-E128 channel order")
        labels_tuple = tuple(labels)
        if channel_labels is None:
            channel_labels = labels_tuple
            encoder = encoder_loader(
                protocol.encoder.checkpoint,
                channel_names=channel_labels,
                mapping_path=Path(mapping_path),
                initialization_seed=protocol.encoder.initialization_seed,
            )
            encoder_sha256 = encoder_state_sha256(encoder)
        elif labels_tuple != channel_labels:
            raise FrozenEncoderError("pilot channel order changed between subjects")
        assert encoder is not None
        assert encoder_sha256 is not None
        if encoder_state_sha256(encoder) != encoder_sha256:
            raise FrozenEncoderError("pilot encoder state changed between subjects")
        representations, evidence = extract_frozen_representations_batched(
            encoder,
            windows_array,
            batch_size=extraction_batch_size,
            device=device,
        )
        if evidence["state_sha256_after"] != encoder_sha256:
            raise FrozenEncoderError("pilot encoder state changed during extraction")
        subject_reports.append(
            {
                "subject_id": subject_id,
                "qc": qc,
                "representation_shapes": {
                    str(index): list(representations[index].shape)
                    for index in PREDECLARED_LAYERS
                },
            }
        )

    assert encoder_sha256 is not None
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "protocol_sha256": protocol.sha256,
        "dataset_manifest_sha256": dataset_sha256,
        "draft_manifest_sha256": _sha256_file(Path(manifest_path)),
        "pilot_subject_list_sha256": _canonical_sha256(pilot),
        "pilot_subject_count": len(pilot),
        "checkpoint": protocol.encoder.checkpoint,
        "checkpoint_state_sha256": encoder_sha256,
        "preprocessing_sha256": preprocessing_contract_sha256(
            protocol.preprocessing
        ),
        "extraction_batch_size": extraction_batch_size,
        "subjects": subject_reports,
    }
    if _contains_forbidden_pilot_field(report):
        raise FrozenEncoderError("pilot report contains forbidden outcome information")
    _write_json_create_only(Path(output_path), report)
    return report


def _resolve_hbn_recording_path(data_root: Path, relative_path: str) -> Path:
    root = Path(data_root).resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FrozenEncoderError(
            f"HBN manifest recording is outside data root: {relative_path}"
        ) from error
    if path.suffix != ".set":
        raise FrozenEncoderError(f"HBN recording is not an EEGLAB .set file: {path}")
    if not path.is_file():
        raise FrozenEncoderError(f"HBN recording does not exist: {path}")
    return path


def _task_from_hbn_path(path: Path) -> str:
    for component in path.stem.split("_"):
        if component.startswith("task-"):
            return component
    raise FrozenEncoderError(f"HBN recording has no task label: {path.name}")


def _hbn_windows(
    prepared: PreparedRecording,
    *,
    contract: PreprocessingContract,
) -> np.ndarray:
    data = np.asarray(prepared.data)
    if data.ndim != 2 or data.shape[0] != len(prepared.channel_names):
        raise FrozenEncoderError("prepared HBN recording has invalid dimensions")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise FrozenEncoderError("prepared HBN recording contains non-finite samples")
    window_samples = round(contract.window_seconds * contract.sample_rate_hz)
    stride_samples = round(contract.stride_seconds * contract.sample_rate_hz)
    maximum_samples = round(
        contract.max_seconds_per_subject * contract.sample_rate_hz
    )
    if window_samples <= 0 or stride_samples <= 0 or maximum_samples < window_samples:
        raise FrozenEncoderError("HBN window contract is invalid")
    if data.shape[1] < maximum_samples:
        raise FrozenEncoderError(
            "prepared HBN recording is shorter than the predeclared 120 seconds"
        )
    bounded = data[:, :maximum_samples]
    starts = range(0, maximum_samples - window_samples + 1, stride_samples)
    windows = np.stack(
        [bounded[:, start : start + window_samples] for start in starts],
        axis=0,
    )
    return np.asarray(windows, dtype=np.float32)


def _required_hbn_channel_order(
    contract: PreprocessingContract,
) -> tuple[str, ...]:
    if (
        contract.mapped_channel_layout != "egi_hydrocel_e1_e128"
        or contract.required_mapped_channels != 128
    ):
        raise FrozenEncoderError("unsupported HBN mapped-channel contract")
    return tuple(f"E{index}" for index in range(1, 129))


def _select_hbn_protocol_channels(
    prepared: PreparedRecording,
    channel_order: tuple[str, ...],
) -> PreparedRecording:
    """Select the exact cross-dataset E1-E128 layout and reject missing channels."""

    data = np.asarray(prepared.data)
    names = tuple(prepared.channel_names)
    if data.ndim != 2 or data.shape[0] != len(names):
        raise FrozenEncoderError("prepared HBN recording has invalid dimensions")
    if len(set(names)) != len(names):
        raise FrozenEncoderError("prepared HBN recording has duplicate channels")
    source_indices = {name: index for index, name in enumerate(names)}
    missing = [name for name in channel_order if name not in source_indices]
    if missing:
        raise FrozenEncoderError(f"missing required HBN channels: {missing}")
    indices = tuple(source_indices[name] for name in channel_order)
    if indices == tuple(range(len(channel_order))) and data.dtype == np.float32:
        selected = data[: len(channel_order)]
    else:
        selected = np.asarray(data[list(indices)], dtype=np.float32)
    return PreparedRecording(selected, channel_order)


def _write_json_exact_resume(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path, "training manifest")
        if existing != dict(payload):
            raise FrozenEncoderError(
                f"existing training manifest does not match this run: {path}"
            )
        return
    try:
        _write_json_create_only(path, payload)
    except FrozenEncoderError:
        existing = _load_json(path, "training manifest")
        if existing != dict(payload):
            raise


def materialize_hbn_representations(
    *,
    protocol: StudyProtocol,
    subject_manifest_path: Path,
    data_root: Path,
    preprocessing_cache_root: Path,
    representation_cache_root: Path,
    training_manifest_path: Path,
    mapping_path: Path,
    repository_root: Path,
    device: str,
    extraction_batch_size: int,
    prepared_loader: Callable[[HbnRecording], PreparedRecording] | None = None,
    encoder_loader: Callable[..., nn.Module] = load_reve_encoder,
) -> dict[str, Any]:
    """Materialize one immutable frozen-REVE cache for HBN train/validation."""

    if device not in {"cpu", "cuda", "mps"}:
        raise FrozenEncoderError("HBN extraction device must be cpu, cuda, or mps")
    if (
        isinstance(extraction_batch_size, bool)
        or not isinstance(extraction_batch_size, int)
        or extraction_batch_size <= 0
    ):
        raise FrozenEncoderError("HBN extraction batch_size must be positive")
    tree_sha256 = source_tree_sha256(Path(repository_root))
    rows = read_manifest(Path(subject_manifest_path))
    if not rows:
        raise FrozenEncoderError("HBN subject manifest is empty")
    for row in rows:
        if row.split not in {"train", "val", "test"}:
            raise FrozenEncoderError(f"HBN manifest has invalid split: {row.split}")
        if (row.release == "R5") != (row.split == "test"):
            raise FrozenEncoderError(
                "HBN R5 must be the sealed test split and no other release may be test"
            )
    selected = [row for row in rows if row.split in {"train", "val"}]
    if {row.split for row in selected} != {"train", "val"}:
        raise FrozenEncoderError("HBN cache requires non-empty train and validation splits")
    subject_ids = [row.subject for row in selected]
    if len(subject_ids) != len(set(subject_ids)):
        raise FrozenEncoderError(
            "HBN train/validation manifest contains duplicate or overlapping subjects"
        )

    recordings: list[HbnRecording] = []
    for row in selected:
        if not np.isfinite(row.age) or not np.isfinite(row.duration_s):
            raise FrozenEncoderError("HBN manifest contains non-finite metadata")
        if row.duration_s < protocol.preprocessing.max_seconds_per_subject:
            raise FrozenEncoderError(
                f"HBN recording is shorter than 120 seconds: {row.subject}"
            )
        path = _resolve_hbn_recording_path(data_root, row.recording_relpath)
        recordings.append(
            HbnRecording(
                path=path,
                release=row.release,
                subject=row.subject,
                task=_task_from_hbn_path(path),
                age=float(row.age),
                duration_s=float(row.duration_s),
            )
        )

    store = PreprocessedRecordingStore(Path(preprocessing_cache_root))
    load_prepared = prepared_loader or store.load
    channel_order = _required_hbn_channel_order(protocol.preprocessing)
    for recording in recordings:
        prepared = load_prepared(recording)
        data = np.asarray(prepared.data)
        if data.ndim != 2 or data.shape[0] != len(prepared.channel_names):
            raise FrozenEncoderError("prepared HBN recording has invalid dimensions")
        _select_hbn_protocol_channels(
            PreparedRecording(
                np.empty((data.shape[0], 0), dtype=np.float32),
                tuple(prepared.channel_names),
            ),
            channel_order,
        )
    encoder = encoder_loader(
        protocol.encoder.checkpoint,
        channel_names=channel_order,
        mapping_path=Path(mapping_path),
        initialization_seed=protocol.encoder.initialization_seed,
    )
    checkpoint_sha256 = encoder_state_sha256(encoder)
    subject_manifest_sha256 = manifest_sha256(Path(subject_manifest_path))
    acquisition_files = [
        {
            "subject_id": recording.subject,
            "path": str(path.relative_to(Path(data_root).resolve())),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for recording in recordings
        for path in hbn_recording_source_files(recording)
    ]
    dataset_sha256 = _canonical_sha256(
        {
            "subject_manifest_sha256": subject_manifest_sha256,
            "acquisition_files": acquisition_files,
        }
    )
    preprocessing_sha256 = preprocessing_contract_sha256(protocol.preprocessing)

    subject_rows = [
        {
            "subject_id": row.subject,
            "split": "validation" if row.split == "val" else "train",
            "age": float(row.age),
        }
        for row in selected
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": protocol.sha256,
        "checkpoint": protocol.encoder.checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256": dataset_sha256,
        "subject_manifest_sha256": subject_manifest_sha256,
        "acquisition_files": acquisition_files,
        "preprocessing_sha256": preprocessing_sha256,
        "source_tree_sha256": tree_sha256,
        "subjects": subject_rows,
    }
    if Path(training_manifest_path).exists():
        existing = _load_json(Path(training_manifest_path), "training manifest")
        if existing != report:
            raise FrozenEncoderError(
                "existing training manifest does not match this run: "
                f"{training_manifest_path}"
            )

    for row, recording in zip(selected, recordings, strict=True):
        identity = RepresentationCacheIdentity(
            protocol_sha256=protocol.sha256,
            checkpoint=protocol.encoder.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            dataset_manifest_sha256=dataset_sha256,
            preprocessing_sha256=preprocessing_sha256,
            subject_id=row.subject,
            source_tree_sha256=tree_sha256,
        )
        cache_entry = Path(representation_cache_root) / identity.key
        if cache_entry.exists():
            load_cached_representations(representation_cache_root, identity)
        else:
            if encoder_state_sha256(encoder) != checkpoint_sha256:
                raise FrozenEncoderError("HBN encoder state changed between subjects")
            prepared = _select_hbn_protocol_channels(
                load_prepared(recording), channel_order
            )
            windows = _hbn_windows(prepared, contract=protocol.preprocessing)
            representations, evidence = extract_frozen_representations_batched(
                encoder,
                windows,
                batch_size=extraction_batch_size,
                device=device,
            )
            if evidence["state_sha256_after"] != checkpoint_sha256:
                raise FrozenEncoderError("HBN encoder state changed during extraction")
            evidence["hbn_qc"] = {
                "subject_id": row.subject,
                "release": row.release,
                "recording_relpath": row.recording_relpath,
                "channel_count": len(channel_order),
                "channel_names": list(channel_order),
                "window_count": int(windows.shape[0]),
                "window_samples": int(windows.shape[2]),
                "cross_block_windows": False,
            }
            write_cached_representations(
                representation_cache_root,
                identity,
                representations,
                evidence=evidence,
            )
    _write_json_exact_resume(Path(training_manifest_path), report)
    return report


class LazyMipdbRepresentationProvider:
    """Create an external cache only after the sealed runner publishes its marker."""

    def __init__(
        self,
        *,
        protocol: StudyProtocol,
        bids_root: Path,
        manifest_path: Path,
        cache_root: Path,
        mapping_path: Path,
        started_marker_path: Path,
        expected_lock_sha256: str,
        device: str,
        extraction_batch_size: int,
        subject_loader: Callable[..., tuple[np.ndarray, dict[str, Any]]] = load_mipdb_resting_subject,
        encoder_loader: Callable[..., nn.Module] = load_reve_encoder,
    ) -> None:
        if device not in {"cpu", "cuda", "mps"}:
            raise FrozenEncoderError("extraction device must be cpu, cuda, or mps")
        if isinstance(extraction_batch_size, bool) or extraction_batch_size <= 0:
            raise FrozenEncoderError("extraction batch_size must be positive")
        manifest = _load_json(manifest_path, "MIPDB manifest")
        if manifest.get("schema_version") != 2 or manifest.get("status") != "finalized":
            raise FrozenEncoderError(
                "external representations require a finalized MIPDB manifest"
            )
        if manifest.get("protocol_sha256") != protocol.sha256:
            raise FrozenEncoderError("MIPDB manifest protocol does not match")
        dataset_sha256 = manifest.get("dataset_manifest_sha256")
        if not isinstance(dataset_sha256, str) or not _is_sha256(dataset_sha256):
            raise FrozenEncoderError("MIPDB manifest dataset identity is invalid")
        raw_subjects = manifest.get("subjects")
        if not isinstance(raw_subjects, list):
            raise FrozenEncoderError("MIPDB manifest subjects must be an array")
        subjects: dict[str, Mapping[str, Any]] = {}
        for item in raw_subjects:
            if not isinstance(item, Mapping) or not isinstance(item.get("subject_id"), str):
                raise FrozenEncoderError("MIPDB manifest has an invalid subject record")
            subject_id = item["subject_id"]
            if subject_id in subjects:
                raise FrozenEncoderError("MIPDB manifest has duplicate subjects")
            subjects[subject_id] = item

        self.protocol = protocol
        self.bids_root = Path(bids_root)
        self.cache_root = Path(cache_root)
        self.mapping_path = Path(mapping_path)
        self.started_marker_path = Path(started_marker_path)
        self.expected_lock_sha256 = expected_lock_sha256
        self.device = device
        self.extraction_batch_size = extraction_batch_size
        self.subject_loader = subject_loader
        self.encoder_loader = encoder_loader
        self.dataset_sha256 = dataset_sha256
        self.preprocessing_sha256 = preprocessing_contract_sha256(
            protocol.preprocessing
        )
        self.subjects = subjects
        if subject_loader is load_mipdb_resting_subject:
            try:
                verify_mipdb_acquisition(Path(bids_root), manifest)
            except MipdbInventoryError as error:
                raise FrozenEncoderError(str(error)) from error
        self._encoder: nn.Module | None = None
        self._channel_labels: tuple[str, ...] | None = None

    def _require_started_marker(self) -> None:
        marker = _load_json(self.started_marker_path, "evaluation started marker")
        if (
            marker.get("lock_sha256") != self.expected_lock_sha256
            or marker.get("state") != "started"
        ):
            raise FrozenEncoderError("evaluation started marker does not match the lock")

    def _validate_identity(self, subject_id: str, identity: RepresentationCacheIdentity) -> None:
        expected = {
            "subject_id": subject_id,
            "protocol_sha256": self.protocol.sha256,
            "checkpoint": self.protocol.encoder.checkpoint,
            "dataset_manifest_sha256": self.dataset_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
        }
        mismatches = [
            field for field, value in expected.items() if getattr(identity, field) != value
        ]
        if mismatches:
            raise FrozenEncoderError(
                "external cache identity mismatch: " + ", ".join(mismatches)
            )

    def __call__(
        self, subject_id: str, identity: RepresentationCacheIdentity
    ) -> ExternalSubjectMaterial:
        self._require_started_marker()
        self._validate_identity(subject_id, identity)
        subject = self.subjects.get(subject_id)
        if subject is None:
            raise FrozenEncoderError("external subject is absent from the MIPDB manifest")

        entry = self.cache_root / identity.key
        if entry.exists():
            return load_cached_external_material(self.cache_root, subject_id, identity)

        windows, raw_qc = self.subject_loader(
            self.bids_root,
            subject,
            contract=self.protocol.preprocessing,
        )
        if _contains_forbidden_pilot_field(raw_qc):
            raise FrozenEncoderError("external QC contains a target, prediction, or metric field")
        qc = dict(raw_qc)
        if qc.get("window_count") != int(np.asarray(windows).shape[0]):
            raise FrozenEncoderError("external QC window count does not match material")
        if "status" in qc and qc["status"] != "passed":
            raise FrozenEncoderError("external QC status is not passed")
        if "subject_id" in qc and qc["subject_id"] != subject_id:
            raise FrozenEncoderError("external QC subject identity does not match")
        if qc.get("qc_reasons") != []:
            raise FrozenEncoderError("external QC contains rejection reasons")
        if (
            qc.get("mapped_channel_count") != 128
            or qc.get("cross_block_windows") is not False
            or qc.get("spatial_interpolation") is not False
        ):
            raise FrozenEncoderError("external QC violates the preprocessing contract")
        qc.update({"status": "passed", "subject_id": subject_id})
        channel_labels = qc.get("channel_labels")
        expected_labels = [f"E{index}" for index in range(1, 129)]
        if not isinstance(channel_labels, list) or channel_labels != expected_labels:
            raise FrozenEncoderError("external QC has an invalid mapped channel inventory")
        labels_tuple = tuple(channel_labels)
        if self._encoder is None:
            self._encoder = self.encoder_loader(
                self.protocol.encoder.checkpoint,
                channel_names=labels_tuple,
                mapping_path=self.mapping_path,
                initialization_seed=self.protocol.encoder.initialization_seed,
            )
            self._channel_labels = labels_tuple
        elif labels_tuple != self._channel_labels:
            raise FrozenEncoderError("external channel order changed between subjects")
        if encoder_state_sha256(self._encoder) != identity.checkpoint_sha256:
            raise FrozenEncoderError(
                "loaded REVE encoder state does not match the sealed checkpoint hash"
            )
        representations, evidence = extract_frozen_representations_batched(
            self._encoder,
            windows,
            batch_size=self.extraction_batch_size,
            device=self.device,
        )
        evidence["external_qc"] = qc
        write_cached_representations(
            self.cache_root,
            identity,
            representations,
            evidence=evidence,
        )
        return ExternalSubjectMaterial(
            representations=representations,
            cache_identity=identity,
            qc=qc,
        )
