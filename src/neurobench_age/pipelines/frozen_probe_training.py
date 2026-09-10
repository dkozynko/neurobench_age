"""Cache-only head training and exact frozen-probe run inventory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import nn

from neurobench_age.heads.math import (
    MeanLinearCopyHead,
    MeanRichStatsResidualHead,
    MultiQueryRichStatsResidualHead,
)
from neurobench_age.research.protocol import StudyProtocol
from neurobench_age.research.training_protocol import FrozenProbeTrainingProtocol

from .frozen_probe import (
    FrozenEncoderError,
    PREDECLARED_LAYERS,
    RepresentationCacheIdentity,
    _canonical_sha256,
    _is_sha256,
    _sha256_file,
    inspect_cached_representation_metadata,
    load_cached_representations,
)

APPROVED_HEADS = (
    "mean_linear",
    "mean_layer_linear",
    "mean_rich_stats_residual",
    "multi_query_rich_stats",
)
_HEAD_LAYERS = {
    "mean_linear": -1,
    "mean_layer_linear": -2,
    "mean_rich_stats_residual": -1,
    "multi_query_rich_stats": -1,
}
DEFAULT_GPU_STAGE_CHUNK_WINDOWS = 1024


def required_layer_for_head(head_name: str) -> int:
    try:
        return _HEAD_LAYERS[head_name]
    except KeyError as error:
        raise FrozenEncoderError(
            f"head must be one of the approved heads: {APPROVED_HEADS}"
        ) from error


def build_frozen_probe_head(
    head_name: str, *, embed_dim: int, n_outputs: int = 1
) -> nn.Module:
    """Build one approved head that accepts cached tokens, never an encoder."""

    required_layer_for_head(head_name)
    if head_name in {"mean_linear", "mean_layer_linear"}:
        return MeanLinearCopyHead(embed_dim=embed_dim, n_outputs=n_outputs)
    if head_name == "mean_rich_stats_residual":
        return MeanRichStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
    return MultiQueryRichStatsResidualHead(
        embed_dim=embed_dim, n_outputs=n_outputs
    )


@dataclass(frozen=True)
class CachedSubjectRecord:
    subject_id: str
    split: str
    age: float
    cache_identity: RepresentationCacheIdentity

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise FrozenEncoderError(
                "frozen-probe records must use train or validation split"
            )
        if not math.isfinite(float(self.age)):
            raise FrozenEncoderError("frozen-probe record must contain a finite age")
        if self.subject_id != self.cache_identity.subject_id:
            raise FrozenEncoderError(
                "record subject_id does not match its cache identity"
            )


@dataclass(frozen=True)
class GlobalWindowBatchPlan:
    """Immutable mapping from global window positions to cached subjects."""

    subject_ids: tuple[str, ...]
    subject_ages: tuple[float, ...]
    subject_offsets: tuple[int, ...]
    references: tuple[tuple[int, int], ...]
    batch_size: int

    @classmethod
    def build(
        cls,
        records: Sequence[CachedSubjectRecord],
        *,
        tensors: Mapping[str, torch.Tensor],
        batch_size: int,
    ) -> GlobalWindowBatchPlan:
        normalized = tuple(records)
        if (
            not normalized
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise FrozenEncoderError("global window batch plan is invalid")
        subject_ids = tuple(record.subject_id for record in normalized)
        if (
            any(
                not isinstance(record, CachedSubjectRecord)
                or record.split != "train"
                for record in normalized
            )
            or len(subject_ids) != len(set(subject_ids))
            or set(tensors) != set(subject_ids)
        ):
            raise FrozenEncoderError("global window batch plan records are invalid")
        expected_tail: tuple[int, ...] | None = None
        expected_dtype: torch.dtype | None = None
        references: list[tuple[int, int]] = []
        subject_offsets = [0]
        for subject_index, subject_id in enumerate(subject_ids):
            tensor = tensors[subject_id]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.ndim != 3
                or tensor.shape[0] <= 0
            ):
                raise FrozenEncoderError(
                    "global window batch plan requires non-empty CPU tensors"
                )
            tail = tuple(tensor.shape[1:])
            if expected_tail is None:
                expected_tail = tail
                expected_dtype = tensor.dtype
            elif tail != expected_tail or tensor.dtype != expected_dtype:
                raise FrozenEncoderError(
                    "global window batch plan tensors are inconsistent"
                )
            references.extend(
                (subject_index, window_index)
                for window_index in range(tensor.shape[0])
            )
            subject_offsets.append(len(references))
        return cls(
            subject_ids=subject_ids,
            subject_ages=tuple(float(record.age) for record in normalized),
            subject_offsets=tuple(subject_offsets),
            references=tuple(references),
            batch_size=batch_size,
        )

    @property
    def total_windows(self) -> int:
        return len(self.references)

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.total_windows / self.batch_size)

    def permuted_batches(
        self, *, generator: torch.Generator
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        shuffled = tuple(
            self.references[index]
            for batch in self.permuted_batch_indices(generator=generator)
            for index in batch.tolist()
        )
        return tuple(
            shuffled[start : start + self.batch_size]
            for start in range(0, len(shuffled), self.batch_size)
        )

    def permuted_batch_indices(
        self, *, generator: torch.Generator
    ) -> tuple[torch.Tensor, ...]:
        """Return the same seeded shuffle as flat CPU row indices."""

        order = torch.randperm(self.total_windows, generator=generator)
        return tuple(
            order[start : start + self.batch_size]
            for start in range(0, self.total_windows, self.batch_size)
        )

    def flatten(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create one contiguous CPU feature/target store for vectorized gathers."""

        if set(tensors) != set(self.subject_ids):
            raise FrozenEncoderError("global window flat store tensors are invalid")
        ordered_tensors = tuple(tensors[subject_id] for subject_id in self.subject_ids)
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.ndim != 3
            or tensor.shape[0] <= 0
            for tensor in ordered_tensors
        ):
            raise FrozenEncoderError("global window flat store requires CPU tensors")
        first_tail = tuple(ordered_tensors[0].shape[1:])
        first_dtype = ordered_tensors[0].dtype
        if any(
            tuple(tensor.shape[1:]) != first_tail or tensor.dtype != first_dtype
            for tensor in ordered_tensors[1:]
        ):
            raise FrozenEncoderError("global window flat store tensors are inconsistent")
        features = torch.cat(ordered_tensors, dim=0)
        targets = torch.cat(
            tuple(
                torch.full(
                    (tensor.shape[0],),
                    age,
                    dtype=features.dtype,
                )
                for tensor, age in zip(ordered_tensors, self.subject_ages)
            ),
            dim=0,
        )
        return features, targets

    def _flat_indices(
        self, references: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        indices: list[int] = []
        for subject_index, window_index in references:
            if (
                isinstance(subject_index, bool)
                or not isinstance(subject_index, int)
                or subject_index < 0
                or subject_index >= len(self.subject_ids)
            ):
                raise FrozenEncoderError("global window subject index is invalid")
            window_count = (
                self.subject_offsets[subject_index + 1]
                - self.subject_offsets[subject_index]
            )
            if (
                isinstance(window_index, bool)
                or not isinstance(window_index, int)
                or window_index < 0
                or window_index >= window_count
            ):
                raise FrozenEncoderError("global window index is invalid")
            indices.append(self.subject_offsets[subject_index] + window_index)
        return torch.tensor(indices, dtype=torch.long)

    def materialize(
        self,
        references: Sequence[tuple[int, int]],
        *,
        tensors: Mapping[str, torch.Tensor] | None = None,
        flat_features: torch.Tensor | None = None,
        flat_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = tuple(references)
        if not normalized or len(normalized) > self.batch_size:
            raise FrozenEncoderError("global window batch references are invalid")
        if (flat_features is None) != (flat_targets is None):
            raise FrozenEncoderError("global window flat store is incomplete")
        if flat_features is not None and flat_targets is not None:
            if (
                flat_features.device.type != "cpu"
                or flat_targets.device.type != "cpu"
                or flat_features.ndim != 3
                or flat_targets.ndim != 1
                or flat_features.shape[0] != self.total_windows
                or flat_targets.shape[0] != self.total_windows
                or flat_targets.dtype != flat_features.dtype
            ):
                raise FrozenEncoderError("global window flat store is invalid")
            indices = self._flat_indices(normalized)
            return (
                flat_features.index_select(0, indices),
                flat_targets.index_select(0, indices),
            )
        if tensors is None:
            raise FrozenEncoderError("global window tensors are required")
        windows: list[torch.Tensor] = []
        ages: list[float] = []
        for subject_index, window_index in normalized:
            if (
                isinstance(subject_index, bool)
                or not isinstance(subject_index, int)
                or subject_index < 0
                or subject_index >= len(self.subject_ids)
            ):
                raise FrozenEncoderError("global window subject index is invalid")
            tensor = tensors.get(self.subject_ids[subject_index])
            if (
                not isinstance(tensor, torch.Tensor)
                or isinstance(window_index, bool)
                or not isinstance(window_index, int)
                or window_index < 0
                or window_index >= tensor.shape[0]
            ):
                raise FrozenEncoderError("global window index is invalid")
            windows.append(tensor[window_index : window_index + 1])
            ages.append(self.subject_ages[subject_index])
        batch = torch.cat(windows, dim=0)
        targets = torch.tensor(ages, dtype=batch.dtype)
        return batch, targets


def validate_training_records(
    records: Sequence[CachedSubjectRecord],
) -> tuple[CachedSubjectRecord, ...]:
    normalized = tuple(records)
    seen: set[str] = set()
    for record in normalized:
        if not isinstance(record, CachedSubjectRecord):
            raise FrozenEncoderError("training records have an invalid item")
        if record.subject_id in seen:
            raise FrozenEncoderError(
                "training records contain duplicate or overlapping subject IDs"
            )
        seen.add(record.subject_id)
    splits = {record.split for record in normalized}
    if splits != {"train", "validation"}:
        raise FrozenEncoderError(
            "training records must contain non-empty train and validation splits"
        )
    return normalized


def _available_memory_bytes(
    *,
    platform_name: str | None = None,
    proc_meminfo_path: Path = Path("/proc/meminfo"),
    vm_stat_output: str | None = None,
) -> int:
    """Return currently available host memory or fail closed."""

    platform_name = sys.platform if platform_name is None else platform_name
    try:
        if platform_name.startswith("linux"):
            lines = Path(proc_meminfo_path).read_text(encoding="utf-8").splitlines()
            value = next(
                int(line.split()[1]) * 1024
                for line in lines
                if line.startswith("MemAvailable:")
            )
        elif platform_name == "darwin":
            output = (
                subprocess.run(
                    ["vm_stat"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                if vm_stat_output is None
                else vm_stat_output
            )
            first, *lines = output.splitlines()
            marker = "page size of "
            page_size = int(first.split(marker, 1)[1].split()[0])
            counts: dict[str, int] = {}
            for line in lines:
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                counts[key.strip()] = int(raw.strip().rstrip("."))
            value = page_size * sum(
                counts[name]
                for name in ("Pages free", "Pages inactive", "Pages speculative")
            )
        else:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if (
                isinstance(pages, bool)
                or not isinstance(pages, int)
                or pages <= 0
                or isinstance(page_size, bool)
                or not isinstance(page_size, int)
                or page_size <= 0
            ):
                raise ValueError("invalid sysconf memory values")
            value = pages * page_size
    except (
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        StopIteration,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        raise FrozenEncoderError("available host memory could not be determined") from error
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FrozenEncoderError("available host memory could not be determined")
    return value


@dataclass(frozen=True)
class ValidatedRepresentationStore:
    """Immutable in-memory view of one fully validated representation cache."""

    _tensors: Mapping[str, Mapping[int, torch.Tensor]]
    projected_bytes: int
    actual_retained_bytes: int
    available_memory_bytes: int
    required_headroom_bytes: int

    @property
    def subject_count(self) -> int:
        return len(self._tensors)

    def tensor(self, subject_id: str, layer: int) -> torch.Tensor:
        try:
            layers = self._tensors[subject_id]
        except KeyError as error:
            raise FrozenEncoderError(
                f"representation store has absent subject: {subject_id}"
            ) from error
        try:
            return layers[int(layer)]
        except KeyError as error:
            raise FrozenEncoderError(
                f"representation store has absent layer {layer} for {subject_id}"
            ) from error

    @classmethod
    def build(
        cls,
        *,
        records: Sequence[CachedSubjectRecord],
        cache_root: Path,
        required_layers: Sequence[int] = PREDECLARED_LAYERS,
        available_memory_bytes: int | None = None,
        metadata_inspector: Callable[..., Mapping[int, Any]] = inspect_cached_representation_metadata,
        strict_loader: Callable[..., dict[int, torch.Tensor]] = load_cached_representations,
    ) -> ValidatedRepresentationStore:
        records = validate_training_records(records)
        required = tuple(int(layer) for layer in required_layers)
        if not required or len(set(required)) != len(required):
            raise FrozenEncoderError("representation store required layers are invalid")
        declared: dict[str, Mapping[int, Any]] = {}
        projected_bytes = 0
        expected_shapes: dict[int, tuple[int, ...]] = {}
        expected_dtypes: dict[int, torch.dtype] = {}
        for record in records:
            metadata = metadata_inspector(
                cache_root, record.cache_identity
            )
            if any(layer not in metadata for layer in required):
                raise FrozenEncoderError("representation metadata is missing a required layer")
            for layer, tensor_metadata in metadata.items():
                if len(tensor_metadata.shape) != 3 or tensor_metadata.shape[0] <= 0:
                    raise FrozenEncoderError(
                        "cached representation must have shape [windows, tokens, features]"
                    )
                suffix = tensor_metadata.shape[1:]
                if layer in expected_shapes and expected_shapes[layer] != suffix:
                    raise FrozenEncoderError(
                        "representation cache tensors have inconsistent token/feature shapes"
                    )
                if layer in expected_dtypes and expected_dtypes[layer] != tensor_metadata.dtype:
                    raise FrozenEncoderError(
                        "representation cache tensors have inconsistent dtypes"
                    )
                expected_shapes[layer] = suffix
                expected_dtypes[layer] = tensor_metadata.dtype
                projected_bytes += tensor_metadata.nbytes
            declared[record.subject_id] = metadata

        available = (
            _available_memory_bytes()
            if available_memory_bytes is None
            else available_memory_bytes
        )
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or available <= 0
        ):
            raise FrozenEncoderError("available host memory must be a positive integer")
        headroom = max(8 * 1024**3, available // 5)
        if projected_bytes + headroom > available:
            raise FrozenEncoderError(
                "insufficient available RAM for validated representation store: "
                f"projected_bytes={projected_bytes} available_bytes={available} "
                f"required_headroom_bytes={headroom}"
            )

        retained: dict[str, Mapping[int, torch.Tensor]] = {}
        actual_bytes = 0
        for record in records:
            tensors = strict_loader(
                cache_root,
                record.cache_identity,
                required_layers=required,
            )
            if tuple(tensors) != required:
                raise FrozenEncoderError("strict cache loader returned unexpected layers")
            for layer in required:
                tensor = tensors[layer]
                expected = declared[record.subject_id][layer]
                if (
                    tensor.device.type != "cpu"
                    or tuple(tensor.shape) != expected.shape
                    or tensor.dtype != expected.dtype
                    or not torch.isfinite(tensor).all()
                ):
                    raise FrozenEncoderError(
                        f"strict cache payload differs from metadata for {record.subject_id} layer {layer}"
                    )
                actual_bytes += tensor.numel() * tensor.element_size()
            retained[record.subject_id] = MappingProxyType(dict(tensors))
        if actual_bytes != projected_bytes:
            raise FrozenEncoderError(
                "retained representation bytes differ from metadata projection"
            )
        return cls(
            _tensors=MappingProxyType(retained),
            projected_bytes=projected_bytes,
            actual_retained_bytes=actual_bytes,
            available_memory_bytes=available,
            required_headroom_bytes=headroom,
        )


def load_frozen_probe_training_manifest(
    path: Path, *, protocol: StudyProtocol
) -> tuple[CachedSubjectRecord, ...]:
    """Load the strict HBN train/validation-to-cache index used by all heads."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenEncoderError(f"could not read training manifest: {path}") from error
    required_fields = {
        "schema_version",
        "protocol_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "subject_manifest_sha256",
        "acquisition_files",
        "preprocessing_sha256",
        "source_tree_sha256",
        "subjects",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise FrozenEncoderError("training manifest fields do not match the strict schema")
    if payload["schema_version"] != 1:
        raise FrozenEncoderError("training manifest schema_version must be 1")
    if payload["protocol_sha256"] != protocol.sha256:
        raise FrozenEncoderError("training manifest protocol_sha256 does not match --protocol")
    if payload["checkpoint"] != protocol.encoder.checkpoint:
        raise FrozenEncoderError("training manifest checkpoint does not match protocol")
    for field in (
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "subject_manifest_sha256",
        "preprocessing_sha256",
        "source_tree_sha256",
    ):
        value = payload[field]
        if not isinstance(value, str) or not _is_sha256(value):
            raise FrozenEncoderError(f"training manifest {field} must be a SHA-256 digest")
    acquisition_files = payload["acquisition_files"]
    if not isinstance(acquisition_files, list) or not acquisition_files:
        raise FrozenEncoderError("training manifest acquisition_files must be non-empty")
    for item in acquisition_files:
        if (
            not isinstance(item, dict)
            or set(item) != {"subject_id", "path", "size_bytes", "sha256"}
            or not isinstance(item["subject_id"], str)
            or not isinstance(item["path"], str)
            or isinstance(item["size_bytes"], bool)
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
            or not _is_sha256(item["sha256"])
        ):
            raise FrozenEncoderError("training manifest acquisition file is invalid")
    expected_dataset_identity = _canonical_sha256(
        {
            "subject_manifest_sha256": payload["subject_manifest_sha256"],
            "acquisition_files": acquisition_files,
        }
    )
    if payload["dataset_manifest_sha256"] != expected_dataset_identity:
        raise FrozenEncoderError("training manifest dataset identity does not match acquisition")
    raw_subjects = payload["subjects"]
    if not isinstance(raw_subjects, list):
        raise FrozenEncoderError("training manifest subjects must be an array")
    records: list[CachedSubjectRecord] = []
    for index, item in enumerate(raw_subjects):
        if not isinstance(item, dict) or set(item) != {"subject_id", "split", "age"}:
            raise FrozenEncoderError(
                f"training manifest subjects[{index}] has invalid fields"
            )
        subject_id = item["subject_id"]
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise FrozenEncoderError(
                f"training manifest subjects[{index}].subject_id is invalid"
            )
        age = item["age"]
        if isinstance(age, bool) or not isinstance(age, (int, float)):
            raise FrozenEncoderError(
                f"training manifest subjects[{index}].age must be numeric"
            )
        identity = RepresentationCacheIdentity(
            protocol_sha256=protocol.sha256,
            checkpoint=payload["checkpoint"],
            checkpoint_sha256=payload["checkpoint_sha256"],
            dataset_manifest_sha256=payload["dataset_manifest_sha256"],
            preprocessing_sha256=payload["preprocessing_sha256"],
            subject_id=subject_id,
            source_tree_sha256=payload["source_tree_sha256"],
        )
        records.append(
            CachedSubjectRecord(
                subject_id=subject_id,
                split=item["split"],
                age=float(age),
                cache_identity=identity,
            )
        )
    return validate_training_records(records)


def predict_cached_subjects(
    head: nn.Module,
    records: Sequence[CachedSubjectRecord],
    *,
    cache_root: Path,
    head_name: str,
    batch_size: int,
    device: str,
    representation_store: ValidatedRepresentationStore | None = None,
    required_layer_resolver: Callable[[str], int] = required_layer_for_head,
) -> tuple[dict[str, float | str], ...]:
    """Predict subject ages from one declared cached layer only."""

    layer_index = required_layer_resolver(head_name)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise FrozenEncoderError("batch_size must be a positive integer")
    if not records:
        raise FrozenEncoderError("prediction records must not be empty")
    subject_ids = [record.subject_id for record in records]
    if len(subject_ids) != len(set(subject_ids)):
        raise FrozenEncoderError("prediction records contain duplicate subject IDs")
    head.to(device)
    head.eval()
    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for record in records:
            cached = (
                representation_store.tensor(record.subject_id, layer_index)
                if representation_store is not None
                else load_cached_representations(
                    cache_root,
                    record.cache_identity,
                    required_layers=(layer_index,),
                )[layer_index]
            )
            if cached.ndim != 3 or cached.shape[0] == 0:
                raise FrozenEncoderError(
                    "cached representation must have shape [windows, tokens, features]"
                )
            window_predictions: list[torch.Tensor] = []
            for start in range(0, cached.shape[0], batch_size):
                batch = cached[start : start + batch_size].to(device)
                prediction = head(batch).reshape(-1)
                if prediction.numel() != batch.shape[0] or not torch.isfinite(prediction).all():
                    raise FrozenEncoderError(
                        "head must produce one finite prediction per cached window"
                    )
                window_predictions.append(prediction.cpu())
            subject_prediction = float(torch.cat(window_predictions).mean())
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "age": float(record.age),
                    "prediction": subject_prediction,
                }
            )
    return tuple(rows)


@dataclass(frozen=True)
class FrozenProbeRunResult:
    manifest: Mapping[str, Any]
    reused: bool


def _strict_training_contract(training: FrozenProbeTrainingProtocol) -> None:
    if training.optimizer.name != "AdamW":
        raise FrozenEncoderError("frozen-probe optimizer must be AdamW")
    if (
        training.batching.unit != "global_window"
        or training.batching.shuffle != "seeded_randperm"
        or training.batching.drop_last
    ):
        raise FrozenEncoderError(
            "frozen-probe training must use seeded global-window batches"
        )
    if (
        training.scheduler.name != "OneCycleLR"
        or training.scheduler.interval != "step"
        or training.scheduler.frequency != 1
        or training.scheduler.anneal_strategy != "cos"
    ):
        raise FrozenEncoderError(
            "frozen-probe scheduler must be step-wise cosine OneCycleLR"
        )
    if training.loss != "MSELoss":
        raise FrozenEncoderError("frozen-probe loss must be MSELoss")
    if training.checkpoint_metric != "validation_pearson" or training.metric_mode != "max":
        raise FrozenEncoderError(
            "frozen-probe checkpoint selection must maximize validation Pearson"
        )
    if (
        training.batch_size <= 0
        or training.max_epochs <= 0
        or training.patience <= 0
        or training.optimizer.learning_rate <= 0
        or training.optimizer.weight_decay < 0
        or training.scheduler.max_learning_rate <= 0
        or training.scheduler.pct_start <= 0
        or training.scheduler.pct_start >= 1
        or training.scheduler.div_factor <= 0
        or training.scheduler.final_div_factor <= 0
        or training.gradient_clip_norm <= 0
        or not _is_sha256(training.sha256)
    ):
        raise FrozenEncoderError("frozen-probe training settings are invalid")


def _run_identity(
    *,
    head_name: str,
    seed: int,
    records: Sequence[CachedSubjectRecord],
    training: FrozenProbeTrainingProtocol,
    training_source_sha256: str,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(training_source_sha256, str) or not _is_sha256(
        training_source_sha256
    ):
        raise FrozenEncoderError("training_source_sha256 must be a SHA-256 digest")
    training_payload = asdict(training)
    training_payload["seeds"] = list(training.seeds)
    cache_contract_fields = (
        "protocol_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "dataset_manifest_sha256",
        "preprocessing_sha256",
        "source_tree_sha256",
    )
    first_identity = records[0].cache_identity
    cache_contract = {
        field: getattr(first_identity, field) for field in cache_contract_fields
    }
    for record in records[1:]:
        actual = {
            field: getattr(record.cache_identity, field)
            for field in cache_contract_fields
        }
        if actual != cache_contract:
            raise FrozenEncoderError(
                "training records contain mixed cache provenance"
            )
    identity = {
        "schema_version": 3,
        "head_name": head_name,
        "seed": seed,
        "representation_protocol_sha256": first_identity.protocol_sha256,
        "training_protocol_sha256": training.sha256,
        "training_source_sha256": training_source_sha256,
        "cache_contract": cache_contract,
        "training": training_payload,
        "subjects": [
            {
                "subject_id": record.subject_id,
                "split": record.split,
                "age": float(record.age),
                "cache_key": record.cache_identity.key,
            }
            for record in records
        ],
    }
    if run_metadata is not None:
        identity["run_metadata"] = dict(run_metadata)
    return identity


def _process_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _prepare_training_store_device(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: str,
    required_headroom_bytes: int = 2 * 1024**3,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Keep the flat store on CPU unless the requested CUDA device can hold it."""

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return features, targets, "cpu"
    required_bytes = (
        features.numel() * features.element_size()
        + targets.numel() * targets.element_size()
    )
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except RuntimeError as error:
        raise FrozenEncoderError("could not inspect available CUDA memory") from error
    if required_bytes + required_headroom_bytes > free_bytes:
        return features, targets, "cpu"
    try:
        return features.to(device), targets.to(device), device
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return features, targets, "cpu"


def _iter_staged_training_batches(
    batch_indices: Sequence[torch.Tensor],
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: str,
    chunk_windows: int = DEFAULT_GPU_STAGE_CHUNK_WINDOWS,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield the original batches while staging bounded CPU chunks to the device.

    The caller still supplies the predeclared seeded batch sequence.  Consecutive
    batches are gathered into one bounded chunk, transferred once, and then
    sliced back into the original batches.  This reduces CPU-to-GPU transfer
    overhead without changing batch membership, order, optimizer steps, or
    scheduler steps.
    """

    if (
        isinstance(chunk_windows, bool)
        or not isinstance(chunk_windows, int)
        or chunk_windows <= 0
    ):
        raise FrozenEncoderError("GPU staging chunk size must be a positive integer")
    if (
        not isinstance(features, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or features.ndim != 3
        or targets.ndim != 1
        or features.shape[0] != targets.shape[0]
        or features.device != targets.device
    ):
        raise FrozenEncoderError("GPU staging feature and target stores are invalid")

    target_device = torch.device(device)
    use_cuda_target = device.startswith("cuda") and torch.cuda.is_available()

    def stage_chunk(
        pending: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joined_indices = torch.cat(tuple(index.to(features.device) for index in pending))
        chunk_features = features.index_select(0, joined_indices)
        chunk_targets = targets.index_select(0, joined_indices)
        if use_cuda_target and chunk_features.device.type == "cpu":
            pinned = False
            try:
                chunk_features = chunk_features.pin_memory()
                chunk_targets = chunk_targets.pin_memory()
                pinned = True
            except RuntimeError:
                # Some containers have a low memlock limit.  Correctness does
                # not depend on pinning, so retain the pageable chunk.
                pass
            chunk_features = chunk_features.to(device, non_blocking=pinned)
            chunk_targets = chunk_targets.to(device, non_blocking=pinned)
        elif chunk_features.device != target_device:
            chunk_features = chunk_features.to(device)
            chunk_targets = chunk_targets.to(device)
        return chunk_features, chunk_targets

    pending: list[torch.Tensor] = []
    pending_windows = 0
    for indices in batch_indices:
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 1
            or indices.numel() <= 0
        ):
            raise FrozenEncoderError("GPU staging batch indices are invalid")
        batch_windows = int(indices.numel())
        if pending and pending_windows + batch_windows > chunk_windows:
            chunk_features, chunk_targets = stage_chunk(pending)
            offset = 0
            for pending_indices in pending:
                count = int(pending_indices.numel())
                yield chunk_features[offset : offset + count], chunk_targets[offset : offset + count]
                offset += count
            pending = []
            pending_windows = 0
        pending.append(indices)
        pending_windows += batch_windows

    if pending:
        chunk_features, chunk_targets = stage_chunk(pending)
        offset = 0
        for pending_indices in pending:
            count = int(pending_indices.numel())
            yield chunk_features[offset : offset + count], chunk_targets[offset : offset + count]
            offset += count


def _finalize_epoch_training_loss(
    epoch_loss_sum: torch.Tensor,
    *,
    epoch_windows: int,
    head_parameters: Sequence[torch.Tensor],
) -> float:
    """Validate one completed epoch with one scalar device synchronization."""

    if epoch_windows <= 0:
        raise FrozenEncoderError("training epoch contained no windows")
    finite_state = torch.isfinite(epoch_loss_sum).all()
    for parameter in head_parameters:
        finite_state = finite_state & torch.isfinite(parameter).all()
    if not bool(finite_state.detach().cpu()):
        raise FrozenEncoderError("training produced a non-finite loss or parameter")
    return float((epoch_loss_sum / epoch_windows).detach().cpu())


def _pearson_from_rows(rows: Sequence[Mapping[str, float | str]]) -> float:
    if len(rows) < 2:
        return float("nan")
    targets = torch.tensor([float(row["age"]) for row in rows], dtype=torch.float64)
    predictions = torch.tensor(
        [float(row["prediction"]) for row in rows], dtype=torch.float64
    )
    targets = targets - targets.mean()
    predictions = predictions - predictions.mean()
    denominator = torch.sqrt(targets.square().sum() * predictions.square().sum())
    if denominator <= 0:
        return float("nan")
    return float((targets * predictions).sum() / denominator)


def _load_completed_run(
    run_dir: Path,
    *,
    expected_identity_sha256: str,
    training_source_sha256: str,
) -> Mapping[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "head_checkpoint.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FrozenEncoderError(f"existing run is incomplete: {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenEncoderError(f"existing run manifest is invalid: {run_dir}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 3
        or manifest.get("status") != "complete"
    ):
        raise FrozenEncoderError(f"existing run completion marker is invalid: {run_dir}")
    claimed_manifest_hash = manifest.get("run_manifest_sha256")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "run_manifest_sha256"
    }
    if claimed_manifest_hash != _canonical_sha256(manifest_body):
        raise FrozenEncoderError(f"existing run manifest hash does not match: {run_dir}")
    if manifest.get("run_identity_sha256") != expected_identity_sha256:
        raise FrozenEncoderError(f"existing run identity does not match: {run_dir}")
    if manifest.get("training_source_sha256") != training_source_sha256:
        raise FrozenEncoderError(f"existing run training source does not match: {run_dir}")
    if manifest.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise FrozenEncoderError(f"existing run checkpoint hash does not match: {run_dir}")
    return manifest


def _configure_strict_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_frozen_probe_run(
    *,
    head_name: str,
    seed: int,
    records: Sequence[CachedSubjectRecord],
    cache_root: Path,
    run_dir: Path,
    training: FrozenProbeTrainingProtocol,
    device: str,
    training_source_sha256: str,
    representation_store: ValidatedRepresentationStore | None = None,
    flat_training_store: tuple[torch.Tensor, torch.Tensor, str] | None = None,
    head_builder: Callable[..., nn.Module] = build_frozen_probe_head,
    required_layer_resolver: Callable[[str], int] = required_layer_for_head,
    run_metadata: Mapping[str, Any] | None = None,
    progress_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> FrozenProbeRunResult:
    """Train one cache-only head and select the earliest best validation epoch."""

    required_layer = required_layer_resolver(head_name)
    records = validate_training_records(records)
    _strict_training_contract(training)
    if seed not in training.seeds:
        raise FrozenEncoderError(f"seed {seed} is outside the predeclared inventory")
    identity = _run_identity(
        head_name=head_name,
        seed=seed,
        records=records,
        training=training,
        training_source_sha256=training_source_sha256,
        run_metadata=run_metadata,
    )
    identity_sha256 = _canonical_sha256(identity)
    run_dir = Path(run_dir)
    if run_dir.exists():
        return FrozenProbeRunResult(
            manifest=_load_completed_run(
                run_dir,
                expected_identity_sha256=identity_sha256,
                training_source_sha256=training_source_sha256,
            ),
            reused=True,
        )

    train_records = tuple(record for record in records if record.split == "train")
    validation_records = tuple(
        record for record in records if record.split == "validation"
    )
    train_tensors: dict[str, torch.Tensor] = {}
    for record in train_records:
        train_tensors[record.subject_id] = (
            representation_store.tensor(record.subject_id, required_layer)
            if representation_store is not None
            else load_cached_representations(
                cache_root,
                record.cache_identity,
                required_layers=(required_layer,),
            )[required_layer]
        )
    batch_plan = GlobalWindowBatchPlan.build(
        train_records,
        tensors=train_tensors,
        batch_size=training.batch_size,
    )
    sample = train_tensors[train_records[0].subject_id]
    if sample.ndim != 3 or sample.shape[-1] <= 0:
        raise FrozenEncoderError(
            "cached representation must have shape [windows, tokens, features]"
        )
    head_embed_dim = int(
        getattr(representation_store, "embed_dim", sample.shape[-1])
    )
    if head_embed_dim <= 0:
        raise FrozenEncoderError("head embedding dimension must be positive")
    if flat_training_store is None:
        train_features, train_targets = batch_plan.flatten(train_tensors)
        train_features, train_targets, train_store_device = (
            _prepare_training_store_device(
                train_features,
                train_targets,
                device=device,
            )
        )
    else:
        if len(flat_training_store) != 3:
            raise FrozenEncoderError("shared flat training store is invalid")
        train_features, train_targets, train_store_device = flat_training_store
        if (
            not isinstance(train_features, torch.Tensor)
            or not isinstance(train_targets, torch.Tensor)
            or not isinstance(train_store_device, str)
            or train_features.ndim != 3
            or train_targets.ndim != 1
            or train_features.shape[0] != batch_plan.total_windows
            or train_targets.shape[0] != batch_plan.total_windows
            or tuple(train_features.shape[1:]) != tuple(sample.shape[1:])
            or train_features.dtype != sample.dtype
            or train_targets.dtype != sample.dtype
            or train_features.device != train_targets.device
            or train_store_device not in {"cpu", device}
        ):
            raise FrozenEncoderError("shared flat training store is invalid")

    _configure_strict_determinism(seed)
    head = head_builder(head_name, embed_dim=head_embed_dim).to(device)
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        head_parameters,
        lr=training.optimizer.learning_rate,
        weight_decay=training.optimizer.weight_decay,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in head_parameters}:
        raise FrozenEncoderError("optimizer does not own exactly the trainable head parameters")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training.scheduler.max_learning_rate,
        epochs=training.max_epochs,
        steps_per_epoch=batch_plan.steps_per_epoch,
        pct_start=training.scheduler.pct_start,
        anneal_strategy=training.scheduler.anneal_strategy,
        div_factor=training.scheduler.div_factor,
        final_div_factor=training.scheduler.final_div_factor,
    )
    loss_function = nn.MSELoss()
    order_generator = torch.Generator(device="cpu").manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    if progress_sink is not None:
        progress_sink(
            {
                "event": "run_start",
                "head_name": head_name,
                "seed": seed,
                "max_epochs": training.max_epochs,
                "total_training_windows": batch_plan.total_windows,
                "steps_per_epoch": batch_plan.steps_per_epoch,
                "training_store_device": train_store_device,
            }
        )
    history: list[dict[str, float | int]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_rows: tuple[dict[str, float | str], ...] | None = None
    epochs_without_improvement = 0
    optimizer_steps = 0
    scheduler_steps = 0

    for epoch_index in range(training.max_epochs):
        head.train()
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_windows = 0
        for batch, targets in _iter_staged_training_batches(
            batch_plan.permuted_batch_indices(generator=order_generator),
            train_features,
            train_targets,
            device=device,
        ):
            optimizer.zero_grad(set_to_none=True)
            predictions = head(batch).reshape(-1)
            loss = loss_function(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head_parameters,
                training.gradient_clip_norm,
            )
            optimizer.step()
            optimizer_steps += 1
            scheduler.step()
            scheduler_steps += 1
            epoch_loss_sum.add_(loss.detach().to(torch.float64) * batch.shape[0])
            epoch_windows += batch.shape[0]

        expected_steps = (epoch_index + 1) * batch_plan.steps_per_epoch
        if optimizer_steps != expected_steps or scheduler_steps != expected_steps:
            raise FrozenEncoderError("optimizer and scheduler step counts drifted")
        epoch_training_mse = _finalize_epoch_training_loss(
            epoch_loss_sum,
            epoch_windows=epoch_windows,
            head_parameters=head_parameters,
        )

        validation_rows = predict_cached_subjects(
            head,
            validation_records,
            cache_root=cache_root,
            head_name=head_name,
            batch_size=training.batch_size,
            device=device,
            representation_store=representation_store,
            required_layer_resolver=required_layer_resolver,
        )
        validation_pearson = _pearson_from_rows(validation_rows)
        validation_mse = sum(
            (float(row["prediction"]) - float(row["age"])) ** 2
            for row in validation_rows
        ) / len(validation_rows)
        history.append(
            {
                "epoch": epoch_index + 1,
                "training_window_mse": epoch_training_mse,
                "validation_subject_mse": validation_mse,
                "validation_subject_pearson": validation_pearson,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        comparable = validation_pearson if math.isfinite(validation_pearson) else -float("inf")
        if comparable > best_score:
            best_score = comparable
            best_epoch = epoch_index + 1
            best_validation_rows = tuple(dict(row) for row in validation_rows)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if progress_sink is not None:
            progress_sink(
                {
                    "event": "epoch_complete",
                    "head_name": head_name,
                    "seed": seed,
                    "epoch": epoch_index + 1,
                    "max_epochs": training.max_epochs,
                    "training_window_mse": history[-1]["training_window_mse"],
                    "validation_subject_mse": validation_mse,
                    "validation_subject_pearson": (
                        validation_pearson
                        if math.isfinite(validation_pearson)
                        else None
                    ),
                    "best_epoch": best_epoch or None,
                    "best_validation_pearson": (
                        best_score if math.isfinite(best_score) else None
                    ),
                    "epochs_without_improvement": epochs_without_improvement,
                    "patience": training.patience,
                    "learning_rate": history[-1]["learning_rate"],
                    "optimizer_steps": optimizer_steps,
                    "scheduler_steps": scheduler_steps,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        if epochs_without_improvement >= training.patience:
            break

    if best_state is None:
        raise FrozenEncoderError("training produced no finite validation Pearson")
    if best_validation_rows is None:
        raise FrozenEncoderError("training produced no selected validation predictions")
    head.load_state_dict(best_state)
    runtime_seconds = time.perf_counter() - started
    peak_accelerator_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0
    )
    parameter_count = sum(parameter.numel() for parameter in head.parameters())

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    transaction_dir = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}-transaction-", dir=run_dir.parent)
    )
    try:
        checkpoint_path = transaction_dir / "head_checkpoint.pt"
        checkpoint_payload = {
                "schema_version": 3,
                "state_dict": best_state,
                "head_name": head_name,
                "seed": seed,
                "selected_epoch": best_epoch,
                "representation_protocol_sha256": records[
                    0
                ].cache_identity.protocol_sha256,
                "training_protocol_sha256": training.sha256,
                "run_identity_sha256": identity_sha256,
                "training_source_sha256": training_source_sha256,
        }
        if run_metadata is not None:
            checkpoint_payload["run_context"] = dict(run_metadata)
        torch.save(checkpoint_payload, checkpoint_path)
        manifest_body: dict[str, Any] = {
            "schema_version": 3,
            "status": "complete",
            "head_name": head_name,
            "layer_index": required_layer,
            "seed": seed,
            "representation_protocol_sha256": records[
                0
            ].cache_identity.protocol_sha256,
            "training_protocol_sha256": training.sha256,
            "training_source_sha256": training_source_sha256,
            "run_identity": identity,
            "run_identity_sha256": identity_sha256,
            "head_parameters": {
                "total": parameter_count,
                "trainable": sum(
                    parameter.numel()
                    for parameter in head.parameters()
                    if parameter.requires_grad
                ),
            },
            "optimizer": {
                "name": training.optimizer.name,
                "learning_rate": training.optimizer.learning_rate,
                "weight_decay": training.optimizer.weight_decay,
                "scheduler": {
                    "name": training.scheduler.name,
                    "max_learning_rate": training.scheduler.max_learning_rate,
                    "pct_start": training.scheduler.pct_start,
                    "anneal_strategy": training.scheduler.anneal_strategy,
                    "div_factor": training.scheduler.div_factor,
                    "final_div_factor": training.scheduler.final_div_factor,
                    "interval": training.scheduler.interval,
                    "frequency": training.scheduler.frequency,
                    "total_steps": batch_plan.steps_per_epoch
                    * training.max_epochs,
                },
            },
            "loss": training.loss,
            "training_unit": "globally_shuffled_window",
            "batching": {
                "batch_size": training.batch_size,
                "drop_last": training.batching.drop_last,
                "shuffle": training.batching.shuffle,
                "total_training_windows": batch_plan.total_windows,
                "steps_per_epoch": batch_plan.steps_per_epoch,
            },
            "optimizer_steps": optimizer_steps,
            "scheduler_steps": scheduler_steps,
            "validation_unit": "subject_arithmetic_mean_of_windows",
            "checkpoint_selection": {
                "metric": training.checkpoint_metric,
                "mode": training.metric_mode,
                "tie_break": "earliest_epoch",
            },
            "selected_epoch": best_epoch,
            "selected_validation_pearson": best_score,
            "validation_history": history,
            "runtime_seconds": runtime_seconds,
            "peak_process_rss_bytes": _process_peak_rss_bytes(),
            "peak_accelerator_memory_bytes": peak_accelerator_memory,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
        }
        if run_metadata is not None:
            manifest_body["run_context"] = dict(run_metadata)
            manifest_body["selected_validation_subject_metrics"] = [
                dict(row) for row in best_validation_rows
            ]
            manifest_body["observed_early_stopping_steps"] = optimizer_steps
        manifest = {
            **manifest_body,
            "run_manifest_sha256": _canonical_sha256(manifest_body),
        }
        (transaction_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        transaction_dir.replace(run_dir)
    except Exception:
        shutil.rmtree(transaction_dir, ignore_errors=True)
        raise
    if progress_sink is not None:
        progress_sink(
            {
                "event": "run_complete",
                "head_name": head_name,
                "seed": seed,
                "selected_epoch": best_epoch,
                "runtime_seconds": runtime_seconds,
            }
        )
    return FrozenProbeRunResult(manifest=manifest, reused=False)


def _expected_run_paths(output_root: Path) -> dict[tuple[str, int], Path]:
    return {
        (head_name, seed): output_root / head_name / f"seed-{seed}"
        for head_name in APPROVED_HEADS
        for seed in range(33, 43)
    }


def audit_frozen_probe_inventory(
    *,
    output_root: Path,
    records: Sequence[CachedSubjectRecord],
    training: FrozenProbeTrainingProtocol,
    training_source_sha256: str,
) -> Mapping[str, Any]:
    """Require and hash exactly four heads by ten predeclared seeds."""

    records = validate_training_records(records)
    _strict_training_contract(training)
    if training.seeds != tuple(range(33, 43)):
        raise FrozenEncoderError("study requires exactly seeds 33 through 42")
    output_root = Path(output_root)
    expected_paths = _expected_run_paths(output_root)
    actual_heads = {
        path.name for path in output_root.iterdir() if path.is_dir()
    } if output_root.is_dir() else set()
    if actual_heads != set(APPROVED_HEADS):
        raise FrozenEncoderError(
            "exact 40-run inventory is required: "
            f"missing_heads={sorted(set(APPROVED_HEADS) - actual_heads)} "
            f"extra_heads={sorted(actual_heads - set(APPROVED_HEADS))}"
        )
    actual_paths = {
        (head_dir.name, run_dir.name)
        for head_dir in output_root.iterdir()
        if head_dir.is_dir()
        for run_dir in head_dir.iterdir()
        if run_dir.is_dir()
    } if output_root.is_dir() else set()
    expected_names = {
        (head_name, f"seed-{seed}") for head_name, seed in expected_paths
    }
    if actual_paths != expected_names:
        raise FrozenEncoderError(
            "exact 40-run inventory is required: "
            f"missing={sorted(expected_names - actual_paths)} "
            f"extra={sorted(actual_paths - expected_names)}"
        )

    runs: list[dict[str, Any]] = []
    representation_protocol_sha256 = records[0].cache_identity.protocol_sha256
    for (head_name, seed), run_dir in expected_paths.items():
        identity_sha256 = _canonical_sha256(
            _run_identity(
                head_name=head_name,
                seed=seed,
                records=records,
                training=training,
                training_source_sha256=training_source_sha256,
            )
        )
        manifest = _load_completed_run(
            run_dir,
            expected_identity_sha256=identity_sha256,
            training_source_sha256=training_source_sha256,
        )
        if manifest.get("head_name") != head_name or manifest.get("seed") != seed:
            raise FrozenEncoderError(f"run identity fields do not match: {run_dir}")
        runs.append(
            {
                "head_name": head_name,
                "seed": seed,
                "representation_protocol_sha256": representation_protocol_sha256,
                "training_protocol_sha256": training.sha256,
                "training_source_sha256": training_source_sha256,
                "run_identity_sha256": identity_sha256,
                "run_manifest_sha256": manifest["run_manifest_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "selected_epoch": manifest["selected_epoch"],
                "head_parameter_count": manifest["head_parameters"]["trainable"],
            }
        )
    inventory_body: dict[str, Any] = {
        "schema_version": 3,
        "status": "complete",
        "representation_protocol_sha256": representation_protocol_sha256,
        "training_protocol_sha256": training.sha256,
        "training_source_sha256": training_source_sha256,
        "heads": list(APPROVED_HEADS),
        "seeds": list(range(33, 43)),
        "run_count": len(runs),
        "runs": runs,
    }
    inventory = {
        **inventory_body,
        "checkpoint_inventory_sha256": _canonical_sha256(inventory_body),
    }
    inventory_path = output_root / "checkpoint_inventory.json"
    if inventory_path.exists():
        try:
            existing = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FrozenEncoderError("checkpoint inventory is unreadable") from error
        if existing != inventory:
            raise FrozenEncoderError("checkpoint inventory does not match the exact runs")
        return existing

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_root,
            prefix=".checkpoint-inventory-",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(inventory, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, inventory_path)
    except FileExistsError as error:
        raise FrozenEncoderError("checkpoint inventory already exists") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return inventory


def train_frozen_probe_study(
    *,
    records: Sequence[CachedSubjectRecord],
    cache_root: Path,
    output_root: Path,
    training: FrozenProbeTrainingProtocol,
    device: str,
    training_source_sha256: str,
    progress_sink: Callable[[Mapping[str, Any]], None] | None = None,
    available_memory_bytes: int | None = None,
    store_metadata_inspector: Callable[..., Mapping[int, Any]] = inspect_cached_representation_metadata,
    store_strict_loader: Callable[..., dict[int, torch.Tensor]] = load_cached_representations,
) -> Mapping[str, Any]:
    """Train or exactly resume the complete predeclared 4x10 run matrix."""

    records = validate_training_records(records)
    if training.seeds != tuple(range(33, 43)):
        raise FrozenEncoderError("study requires exactly seeds 33 through 42")
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise FrozenEncoderError("frozen-probe output_root must be absolute")
    _strict_training_contract(training)
    expected_paths = _expected_run_paths(output_root)
    actual_heads = (
        {path.name for path in output_root.iterdir() if path.is_dir()}
        if output_root.is_dir()
        else set()
    )
    unexpected_heads = actual_heads - set(APPROVED_HEADS)
    if unexpected_heads:
        raise FrozenEncoderError(
            f"study preflight found unexpected head directories: {sorted(unexpected_heads)}"
        )
    unexpected_runs = {
        (head_dir.name, run_dir.name)
        for head_dir in output_root.iterdir()
        if head_dir.is_dir() and head_dir.name in APPROVED_HEADS
        for run_dir in head_dir.iterdir()
        if run_dir.is_dir()
        and (head_dir.name, run_dir.name)
        not in {
            (head_name, f"seed-{seed}")
            for head_name, seed in expected_paths
        }
    } if output_root.is_dir() else set()
    if unexpected_runs:
        raise FrozenEncoderError(
            f"study preflight found unexpected run directories: {sorted(unexpected_runs)}"
        )
    for (head_name, seed), run_dir in expected_paths.items():
        if not run_dir.exists():
            continue
        identity_sha256 = _canonical_sha256(
            _run_identity(
                head_name=head_name,
                seed=seed,
                records=records,
                training=training,
                training_source_sha256=training_source_sha256,
            )
        )
        _load_completed_run(
            run_dir,
            expected_identity_sha256=identity_sha256,
            training_source_sha256=training_source_sha256,
        )
    inventory_path = output_root / "checkpoint_inventory.json"
    if inventory_path.exists() and any(
        not path.is_dir() for path in expected_paths.values()
    ):
        raise FrozenEncoderError(
            "study preflight found checkpoint inventory before all runs are complete"
        )
    if progress_sink is not None:
        progress_sink(
            {
                "event": "store_load_start",
                "subject_count": len(records),
            }
        )
    store = ValidatedRepresentationStore.build(
        records=records,
        cache_root=cache_root,
        available_memory_bytes=available_memory_bytes,
        metadata_inspector=store_metadata_inspector,
        strict_loader=store_strict_loader,
    )
    if progress_sink is not None:
        progress_sink(
            {
                "event": "store_load_complete",
                "subject_count": store.subject_count,
                "projected_bytes": store.projected_bytes,
                "actual_retained_bytes": store.actual_retained_bytes,
                "available_memory_bytes": store.available_memory_bytes,
                "required_headroom_bytes": store.required_headroom_bytes,
            }
        )
    completed = 0
    shared_cpu_stores: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    active_layer: int | None = None
    active_store: tuple[torch.Tensor, torch.Tensor, str] | None = None
    for head_name in APPROVED_HEADS:
        for seed in training.seeds:
            required_layer = required_layer_for_head(head_name)
            if required_layer not in shared_cpu_stores:
                train_records = tuple(
                    record for record in records if record.split == "train"
                )
                tensors = {
                    record.subject_id: store.tensor(record.subject_id, required_layer)
                    for record in train_records
                }
                plan = GlobalWindowBatchPlan.build(
                    train_records,
                    tensors=tensors,
                    batch_size=training.batch_size,
                )
                shared_cpu_stores[required_layer] = plan.flatten(tensors)
            if active_layer != required_layer:
                if active_store is not None and active_store[0].device.type == "cuda":
                    del active_store
                    torch.cuda.empty_cache()
                cpu_features, cpu_targets = shared_cpu_stores[required_layer]
                active_store = (
                    *_prepare_training_store_device(
                        cpu_features,
                        cpu_targets,
                        device=device,
                    ),
                )
                active_layer = required_layer
            result = train_frozen_probe_run(
                head_name=head_name,
                seed=seed,
                records=records,
                cache_root=cache_root,
                run_dir=output_root / head_name / f"seed-{seed}",
                training=training,
                device=device,
                training_source_sha256=training_source_sha256,
                representation_store=store,
                flat_training_store=active_store,
                progress_sink=progress_sink,
            )
            completed += 1
            if progress_sink is not None:
                progress_sink(
                    {
                        "event": "study_progress",
                        "completed_runs": completed,
                        "total_runs": 40,
                        "head_name": head_name,
                        "seed": seed,
                        "reused": result.reused,
                    }
                )
    inventory = audit_frozen_probe_inventory(
        output_root=output_root,
        records=records,
        training=training,
        training_source_sha256=training_source_sha256,
    )
    if progress_sink is not None:
        progress_sink(
            {
                "event": "study_complete",
                "run_count": inventory["run_count"],
                "checkpoint_inventory_sha256": inventory[
                    "checkpoint_inventory_sha256"
                ],
            }
        )
    return inventory
