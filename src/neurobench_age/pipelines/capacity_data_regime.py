"""Extension-only heads and orchestration for the capacity--data analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from neurobench_age.heads.math import (
    MeanLinearCopyHead,
    MeanMLPResidualHead,
    MeanRichStatsResidualHead,
    head_complexity_metadata,
)
from neurobench_age.research.capacity_data_regime import (
    CapacityDataRegimeProtocol,
    canonical_cache_manifest,
)
from neurobench_age.research.capacity_data_regime_lock import (
    build_checkpoint_inventory,
    build_checkpoint_sealed_lock,
    build_lock_core,
    create_lifecycle_sidecar,
    transition_lifecycle,
)
from neurobench_age.research.training_protocol import FrozenProbeTrainingProtocol
from .frozen_probe import FrozenEncoderError
from .frozen_probe_training import (
    CachedSubjectRecord,
    DEFAULT_GPU_STAGE_CHUNK_WINDOWS,
    FrozenProbeRunResult,
    GlobalWindowBatchPlan,
    ValidatedRepresentationStore,
    _prepare_training_store_device,
    train_frozen_probe_run,
)


class CapacityDataRegimePipelineError(RuntimeError):
    """Raised when extension preflight or execution invariants fail."""


CAPACITY_HEADS = (
    "mean_linear",
    "mean_rich_stats_residual",
    "mean_mlp_residual_matched(hidden_dim=4)",
)


def capacity_summary_mode(head_name: str) -> str:
    """Return the fixed per-window summary required by one capacity head."""

    _validate_capacity_head_name(head_name)
    return "rich_stats" if head_name == "mean_rich_stats_residual" else "mean"


def capacity_summary_from_tokens(
    tokens: torch.Tensor,
    *,
    head_name: str,
    embed_dim: int,
) -> torch.Tensor:
    """Convert token sequences to a head-equivalent one-token summary.

    The summary is computed before optimization and contains the same fixed
    statistics used by the capacity heads.  The leading singleton token axis
    keeps the cached tensor contract ``[windows, tokens, features]`` intact.
    """

    mode = capacity_summary_mode(head_name)
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 3
        or tokens.shape[1] <= 0
        or tokens.shape[-1] != embed_dim
    ):
        raise CapacityDataRegimePipelineError(
            "capacity summary requires [windows, tokens, embed_dim] tensors"
        )
    mean = tokens.mean(dim=1)
    if mode == "mean":
        return mean.unsqueeze(1)

    stats_mean = tokens.mean(dim=1, keepdim=True)
    standard_deviation = tokens.std(dim=1, unbiased=False)
    value_range = tokens.amax(dim=1) - tokens.amin(dim=1)
    mean_absolute_deviation = (tokens - stats_mean).abs().mean(dim=1)
    mean_absolute_value = tokens.abs().mean(dim=1)
    return torch.cat(
        (mean, standard_deviation, value_range, mean_absolute_deviation, mean_absolute_value),
        dim=-1,
    ).unsqueeze(1)


@dataclass(frozen=True)
class CapacitySummaryRepresentationStore:
    """Immutable CPU store of fixed per-window capacity summaries."""

    tensors: Mapping[str, torch.Tensor]
    embed_dim: int
    summary_mode: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.embed_dim, bool)
            or not isinstance(self.embed_dim, int)
            or self.embed_dim <= 0
            or self.summary_mode not in {"mean", "rich_stats"}
            or not self.tensors
        ):
            raise CapacityDataRegimePipelineError("capacity summary store metadata is invalid")
        expected_width = self.embed_dim * (5 if self.summary_mode == "rich_stats" else 1)
        normalized = dict(self.tensors)
        for subject_id, tensor in normalized.items():
            if (
                not isinstance(subject_id, str)
                or not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.ndim != 3
                or tensor.shape[0] <= 0
                or tensor.shape[1] != 1
                or tensor.shape[2] != expected_width
                or not torch.isfinite(tensor).all()
            ):
                raise CapacityDataRegimePipelineError(
                    "capacity summary store tensors are invalid"
                )
        object.__setattr__(self, "tensors", MappingProxyType(normalized))

    @property
    def subject_count(self) -> int:
        return len(self.tensors)

    def tensor(self, subject_id: str, layer: int) -> torch.Tensor:
        if layer != -1:
            raise CapacityDataRegimePipelineError(
                "capacity summary store exposes only the final-layer summary"
            )
        try:
            return self.tensors[subject_id]
        except KeyError as error:
            raise CapacityDataRegimePipelineError(
                f"capacity summary store is missing subject: {subject_id}"
            ) from error


def build_capacity_summary_store(
    *,
    records: Sequence[CachedSubjectRecord],
    representation_store: Any,
    head_name: str,
    compute_device: str,
) -> CapacitySummaryRepresentationStore:
    """Build one reusable summary store from the validated raw cache."""

    if not records or representation_store is None:
        raise CapacityDataRegimePipelineError("capacity summary inputs are incomplete")
    mode = capacity_summary_mode(head_name)
    first_tensor = representation_store.tensor(records[0].subject_id, -1)
    if not isinstance(first_tensor, torch.Tensor) or first_tensor.ndim != 3:
        raise CapacityDataRegimePipelineError("raw capacity representation is invalid")
    embed_dim = int(first_tensor.shape[-1])
    target_device = torch.device(compute_device)
    summaries: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for record in records:
            if record.subject_id in summaries:
                raise CapacityDataRegimePipelineError(
                    "capacity summary records contain duplicate subject IDs"
                )
            tokens = representation_store.tensor(record.subject_id, -1)
            if tokens.device.type != target_device.type:
                tokens = tokens.to(target_device)
            summary = capacity_summary_from_tokens(
                tokens,
                head_name=head_name,
                embed_dim=embed_dim,
            )
            summaries[record.subject_id] = summary.detach().cpu()
    return CapacitySummaryRepresentationStore(
        tensors=summaries,
        embed_dim=embed_dim,
        summary_mode=mode,
    )


@dataclass(frozen=True)
class CapacityRunIdentity:
    training_size: int
    head: str
    seed: int


def extension_run_matrix(
    protocol: CapacityDataRegimeProtocol,
) -> tuple[CapacityRunIdentity, ...]:
    """Return the exact predeclared size-by-head-by-seed matrix."""

    if tuple(protocol.head_names) != CAPACITY_HEADS:
        raise CapacityDataRegimePipelineError(
            "protocol heads do not match the exact extension allowlist"
        )
    matrix = tuple(
        CapacityRunIdentity(training_size=size, head=head, seed=seed)
        for size in protocol.training_sizes
        for head in protocol.head_names
        for seed in protocol.seeds
    )
    if len(matrix) != 90 or len(set(matrix)) != 90:
        raise CapacityDataRegimePipelineError(
            "extension matrix must contain exactly 90 unique identities"
        )
    return matrix


def capacity_free_space_requirement(
    *,
    representation_cache_bytes: int,
    estimated_extension_output_bytes: int,
    free_space_floor_bytes: int = 50 * 1024**3,
    free_space_cache_fraction: float = 0.25,
    free_space_output_multiplier: int = 2,
) -> int:
    """Compute the fail-closed free-space threshold from the frozen contract."""

    if (
        isinstance(representation_cache_bytes, bool)
        or not isinstance(representation_cache_bytes, int)
        or representation_cache_bytes < 0
        or isinstance(estimated_extension_output_bytes, bool)
        or not isinstance(estimated_extension_output_bytes, int)
        or estimated_extension_output_bytes < 0
        or isinstance(free_space_floor_bytes, bool)
        or not isinstance(free_space_floor_bytes, int)
        or free_space_floor_bytes < 0
        or isinstance(free_space_cache_fraction, bool)
        or not isinstance(free_space_cache_fraction, (int, float))
        or not math.isfinite(float(free_space_cache_fraction))
        or free_space_cache_fraction < 0
        or isinstance(free_space_output_multiplier, bool)
        or not isinstance(free_space_output_multiplier, int)
        or free_space_output_multiplier < 0
    ):
        raise CapacityDataRegimePipelineError("preflight byte values are invalid")
    return max(
        free_space_floor_bytes,
        math.ceil(free_space_cache_fraction * representation_cache_bytes),
    ) + free_space_output_multiplier * estimated_extension_output_bytes


def compute_capacity_preflight(
    *,
    representation_cache_bytes: int,
    estimated_extension_output_bytes: int,
    free_space_bytes: int,
    available_memory_bytes: int,
    projected_store_bytes: int,
    train_windows: int,
    pilot_seconds: float,
    cached_window_count: int,
    free_space_floor_bytes: int = 50 * 1024**3,
    free_space_cache_fraction: float = 0.25,
    free_space_output_multiplier: int = 2,
) -> dict[str, int | float]:
    """Validate storage/RAM headroom and return recorded preflight evidence."""

    required = capacity_free_space_requirement(
        representation_cache_bytes=representation_cache_bytes,
        estimated_extension_output_bytes=estimated_extension_output_bytes,
        free_space_floor_bytes=free_space_floor_bytes,
        free_space_cache_fraction=free_space_cache_fraction,
        free_space_output_multiplier=free_space_output_multiplier,
    )
    values = {
        "free_space_bytes": free_space_bytes,
        "available_memory_bytes": available_memory_bytes,
        "projected_store_bytes": projected_store_bytes,
        "train_windows": train_windows,
        "cached_window_count": cached_window_count,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values.values()
    ):
        raise CapacityDataRegimePipelineError("preflight measured values are invalid")
    if isinstance(pilot_seconds, bool) or not isinstance(pilot_seconds, (int, float)) or pilot_seconds < 0:
        raise CapacityDataRegimePipelineError("pilot_seconds is invalid")
    if free_space_bytes < required:
        raise CapacityDataRegimePipelineError(
            f"insufficient free space: required={required} available={free_space_bytes}"
        )
    if projected_store_bytes > available_memory_bytes:
        raise CapacityDataRegimePipelineError(
            "cache loading would duplicate the full cache beyond available memory"
        )
    if train_windows <= 0 or cached_window_count <= 0:
        raise CapacityDataRegimePipelineError("preflight window counts must be positive")
    return {
        "representation_cache_bytes": representation_cache_bytes,
        "estimated_extension_output_bytes": estimated_extension_output_bytes,
        "free_space_bytes": free_space_bytes,
        "required_free_space_bytes": required,
        "peak_ram_bytes": projected_store_bytes,
        "upper_bound_optimizer_steps": 40 * math.ceil(train_windows / 64),
        "observed_pilot_seconds": float(pilot_seconds),
        "cached_window_count": cached_window_count,
    }


def validate_extension_output_root(
    output_root: Path,
    *,
    repository_root: Path,
    raw_data_roots: tuple[Path, ...],
    primary_evidence_roots: tuple[Path, ...],
) -> None:
    """Reject extension output locations that could mix with primary evidence."""

    output = Path(output_root)
    if not output.is_absolute():
        raise CapacityDataRegimePipelineError("extension output root must be absolute")
    resolved_output = output.resolve()
    roots = (("repository", Path(repository_root)), *[("raw data", root) for root in raw_data_roots], *[("primary evidence", root) for root in primary_evidence_roots])
    for label, root in roots:
        if resolved_output == root.resolve() or resolved_output.is_relative_to(root.resolve()):
            raise CapacityDataRegimePipelineError(
                f"extension output root must be outside {label}: {resolved_output}"
            )


def select_capacity_records(
    records: Any, ordered_train_subject_ids: tuple[str, ...]
) -> tuple[Any, ...]:
    """Select one nested train cohort while retaining the fixed validation set."""

    by_subject: dict[str, Any] = {}
    for record in records:
        subject_id = getattr(record, "subject_id", None)
        if not isinstance(subject_id, str) or not subject_id:
            raise CapacityDataRegimePipelineError("training record subject_id is invalid")
        if subject_id in by_subject:
            raise CapacityDataRegimePipelineError(
                "training records contain duplicate subject IDs"
            )
        by_subject[subject_id] = record
    if len(set(ordered_train_subject_ids)) != len(ordered_train_subject_ids):
        raise CapacityDataRegimePipelineError("nested cohort contains duplicate subject IDs")
    missing = [subject_id for subject_id in ordered_train_subject_ids if subject_id not in by_subject]
    if missing:
        raise CapacityDataRegimePipelineError(
            f"nested cohort subjects are absent from training manifest: {missing[:3]}"
        )
    train_records = tuple(by_subject[subject_id] for subject_id in ordered_train_subject_ids)
    if any(getattr(record, "split", None) != "train" for record in train_records):
        raise CapacityDataRegimePipelineError("nested cohort contains a non-train subject")
    validation_records = tuple(
        record
        for subject_id, record in sorted(by_subject.items(), key=lambda item: item[0].encode("utf-8"))
        if getattr(record, "split", None) == "validation"
    )
    if not validation_records:
        raise CapacityDataRegimePipelineError("capacity cohort requires validation records")
    if set(record.subject_id for record in train_records) & {
        record.subject_id for record in validation_records
    }:
        raise CapacityDataRegimePipelineError("nested cohort overlaps validation subjects")
    return train_records + validation_records


def build_cache_manifest_from_records(
    records: Sequence[CachedSubjectRecord], *, cache_root: Path
) -> tuple[tuple[dict[str, Any], ...], str, int, int, int]:
    """Inspect every cached layer and return manifest, hash, bytes, and windows."""

    rows: list[dict[str, Any]] = []
    cache_bytes = 0
    cached_window_count = 0
    train_window_count = 0
    for record in records:
        entry = Path(cache_root) / record.cache_identity.key
        metadata_path = entry / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CapacityDataRegimePipelineError(
                f"cache metadata is unreadable: {metadata_path}"
            ) from error
        if not isinstance(metadata, Mapping) or metadata.get("cache_key") != record.cache_identity.key:
            raise CapacityDataRegimePipelineError(
                f"cache metadata identity does not match record: {entry}"
            )
        evidence = metadata.get("evidence")
        hbn_qc = evidence.get("hbn_qc") if isinstance(evidence, Mapping) else None
        if not isinstance(hbn_qc, Mapping) or not isinstance(hbn_qc.get("recording_relpath"), str):
            raise CapacityDataRegimePipelineError(
                f"cache metadata is missing HBN recording provenance: {metadata_path}"
            )
        tensor_metadata = metadata.get("tensor_metadata")
        if not isinstance(tensor_metadata, Mapping):
            raise CapacityDataRegimePipelineError(
                f"cache tensor metadata is invalid: {metadata_path}"
            )
        window_count = None
        for layer_index in (-2, -1):
            raw = tensor_metadata.get(str(layer_index))
            if not isinstance(raw, Mapping) or not isinstance(raw.get("shape"), list):
                raise CapacityDataRegimePipelineError(
                    f"cache layer metadata is invalid: {metadata_path}"
                )
            shape = raw["shape"]
            if not shape or not isinstance(shape[0], int) or shape[0] <= 0:
                raise CapacityDataRegimePipelineError(
                    f"cache window metadata is invalid: {metadata_path}"
                )
            if window_count is None:
                window_count = int(shape[0])
            elif window_count != int(shape[0]):
                raise CapacityDataRegimePipelineError(
                    f"cache layers disagree on window count: {metadata_path}"
                )
            dtype_name = raw.get("dtype")
            if not isinstance(dtype_name, str):
                raise CapacityDataRegimePipelineError(
                    f"cache dtype metadata is invalid: {metadata_path}"
                )
            dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(dtype_name)
            if dtype is None:
                raise CapacityDataRegimePipelineError(
                    f"cache dtype is not supported by extension manifest: {dtype_name}"
                )
            cache_bytes += math.prod(int(value) for value in shape) * torch.empty((), dtype=dtype).element_size()
            rows.append(
                {
                    "subject_id": record.subject_id,
                    "split": record.split,
                    "recording_window_key": f"{hbn_qc['recording_relpath']}#windows-{window_count:06d}",
                    "cache_key": record.cache_identity.key,
                    "layer_index": layer_index,
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "payload_sha256": metadata.get("payload_sha256"),
                }
            )
        assert window_count is not None
        cached_window_count += window_count
        if record.split == "train":
            train_window_count += window_count
    canonical, digest = canonical_cache_manifest(rows)
    return canonical, digest, cache_bytes, cached_window_count, train_window_count


def required_capacity_layer_for_head(head_name: str) -> int:
    _validate_capacity_head_name(head_name)
    return -1


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise CapacityDataRegimePipelineError(f"could not hash artifact: {path}") from error


def snapshot_primary_artifacts(
    *, primary_study_lock: Path, primary_prediction_inventory: Path
) -> dict[str, str]:
    """Hash primary evidence before extension execution."""

    paths = {
        "primary_study_lock_sha256": Path(primary_study_lock),
        "primary_prediction_inventory_sha256": Path(primary_prediction_inventory),
    }
    return {field: _sha256_file(path) for field, path in paths.items()}


def _assert_primary_snapshot_unchanged(
    *,
    before: Mapping[str, str],
    primary_study_lock: Path,
    primary_prediction_inventory: Path,
) -> None:
    after = snapshot_primary_artifacts(
        primary_study_lock=primary_study_lock,
        primary_prediction_inventory=primary_prediction_inventory,
    )
    if dict(before) != after:
        raise CapacityDataRegimePipelineError(
            "primary study lock or prediction inventory changed during extension"
        )


def _assert_no_external_artifacts(output_root: Path) -> None:
    if not output_root.exists():
        return
    forbidden = {"mipdb", "prediction", "predictions", "holdout"}
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        if any(token in path.name.casefold() for token in forbidden):
            raise CapacityDataRegimePipelineError(
                f"MIPDB artifact is not allowed in training or selection output: {path}"
            )


def _validate_capacity_output_layout(output_root: Path) -> None:
    """Reject unexpected files/directories before a resume can be attempted."""

    allowed_top_level = {
        "lifecycle",
        "n-200",
        "n-400",
        "n-800",
        "checkpoint_inventory.json",
        "checkpoint_sealed_lock.json",
    }
    for child in output_root.iterdir():
        if child.name not in allowed_top_level:
            raise CapacityDataRegimePipelineError(
                f"unexpected extension output artifact: {child}"
            )
    expected_heads = {_head_directory_name(head) for head in CAPACITY_HEADS}
    for size in (200, 400, 800):
        size_root = output_root / f"n-{size}"
        if not size_root.exists():
            continue
        if not size_root.is_dir():
            raise CapacityDataRegimePipelineError(
                f"extension size output is not a directory: {size_root}"
            )
        for head_dir in size_root.iterdir():
            if head_dir.name not in expected_heads or not head_dir.is_dir():
                raise CapacityDataRegimePipelineError(
                    f"unexpected extension head output: {head_dir}"
                )
            for run_dir in head_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.startswith("seed-"):
                    raise CapacityDataRegimePipelineError(
                        f"unexpected extension run output: {run_dir}"
                    )
                actual_files = {path.name for path in run_dir.iterdir() if path.is_file()}
                if actual_files and actual_files != {"run_manifest.json", "head_checkpoint.pt"}:
                    raise CapacityDataRegimePipelineError(
                        f"extension run contains unexpected files: {run_dir}"
                    )


def _write_json_exact(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CapacityDataRegimePipelineError(
                f"existing extension artifact is unreadable: {path}"
            ) from error
        if existing != dict(payload):
            raise CapacityDataRegimePipelineError(
                f"existing extension artifact does not match: {path}"
            )
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise CapacityDataRegimePipelineError(
                f"extension artifact already exists: {path}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _head_directory_name(head_name: str) -> str:
    return {
        "mean_linear": "mean_linear",
        "mean_rich_stats_residual": "mean_rich_stats_residual",
        "mean_mlp_residual_matched(hidden_dim=4)": "mean_mlp_residual_matched_h4",
    }[head_name]


def _capacity_run_record(
    identity: CapacityRunIdentity, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("status") != "complete":
        raise CapacityDataRegimePipelineError("extension run is not complete")
    if manifest.get("head_name") != identity.head or manifest.get("seed") != identity.seed:
        raise CapacityDataRegimePipelineError("extension run identity fields do not match")
    run_context = manifest.get("run_context")
    if not isinstance(run_context, Mapping):
        raise CapacityDataRegimePipelineError("extension run is missing run_context")
    if (
        run_context.get("extension_id") != "reve_age_capacity_data_regime_v1"
        or run_context.get("training_size") != identity.training_size
        or run_context.get("head") != identity.head
    ):
        raise CapacityDataRegimePipelineError("extension run context does not match identity")
    complexity = run_context.get("head_complexity")
    resource = run_context.get("resource")
    if not isinstance(complexity, Mapping) or not isinstance(resource, Mapping):
        raise CapacityDataRegimePipelineError(
            "extension run is missing head complexity or resource metadata"
        )
    batching = manifest.get("batching")
    history = manifest.get("validation_history")
    selected_metrics = manifest.get("selected_validation_subject_metrics")
    if (
        not isinstance(batching, Mapping)
        or not isinstance(batching.get("total_training_windows"), int)
        or batching["total_training_windows"] <= 0
        or not isinstance(manifest.get("optimizer_steps"), int)
        or manifest["optimizer_steps"] <= 0
        or not isinstance(manifest.get("observed_early_stopping_steps"), int)
        or manifest["observed_early_stopping_steps"] <= 0
        or not isinstance(history, list)
        or not history
        or not isinstance(selected_metrics, list)
        or not selected_metrics
        or not isinstance(manifest.get("selected_epoch"), int)
        or manifest["selected_epoch"] <= 0
    ):
        raise CapacityDataRegimePipelineError(
            "extension run is missing complete training observations"
        )
    run_manifest_sha256 = manifest.get("run_manifest_sha256")
    checkpoint_sha256 = manifest.get("checkpoint_sha256")
    if (
        not isinstance(run_manifest_sha256, str)
        or len(run_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in run_manifest_sha256)
    ):
        raise CapacityDataRegimePipelineError("extension run manifest digest is invalid")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise CapacityDataRegimePipelineError("extension checkpoint digest is invalid")
    return {
        "training_size": identity.training_size,
        "head": identity.head,
        "seed": identity.seed,
        "run_manifest_sha256": run_manifest_sha256,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "status": "complete",
        "cached_window_count": batching["total_training_windows"],
        "optimizer_steps": manifest["optimizer_steps"],
        "observed_early_stopping_steps": manifest[
            "observed_early_stopping_steps"
        ],
        "validation_history": history,
        "selected_validation_subject_metrics": selected_metrics,
        "selected_epoch": manifest["selected_epoch"],
        "head_complexity": dict(complexity),
        "resource": dict(resource),
    }


def validate_primary_run_reuse(
    manifest: Mapping[str, Any], *, expected_fields: Mapping[str, Any]
) -> bool:
    """Accept a primary run for reuse only when its identity is exact.

    Capacity--data runs may reuse an existing primary ``n=800`` checkpoint,
    but only after the complete run manifest has been compared against the
    predeclared identity.  This helper intentionally performs equality checks
    on the caller-supplied fields rather than accepting a looser
    "compatible-looking" manifest.
    """

    if not isinstance(manifest, Mapping) or manifest.get("status") != "complete":
        raise CapacityDataRegimePipelineError(
            "exact primary reuse requires a complete run manifest"
        )
    if not expected_fields:
        raise CapacityDataRegimePipelineError(
            "exact primary reuse requires predeclared identity fields"
        )
    for field, expected in expected_fields.items():
        if field not in manifest or manifest[field] != expected:
            raise CapacityDataRegimePipelineError(
                f"exact primary reuse identity mismatch for field: {field}"
            )
    return True


def run_capacity_data_regime(
    *,
    protocol: CapacityDataRegimeProtocol,
    training: FrozenProbeTrainingProtocol,
    records: Sequence[CachedSubjectRecord],
    cohorts: Mapping[int, Sequence[str]],
    cache_root: Path,
    output_root: Path,
    repository_root: Path,
    raw_data_roots: tuple[Path, ...],
    primary_evidence_roots: tuple[Path, ...],
    primary_study_lock: Path,
    primary_prediction_inventory: Path,
    training_source_sha256: str,
    core_bundle: Mapping[str, Any],
    preflight_report: Mapping[str, int | float],
    device: str,
    available_memory_bytes: int | None = None,
    representation_store: Any | None = None,
    run_callable: Callable[..., FrozenProbeRunResult] | None = None,
    progress_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    """Train/resume and seal the complete 3x3x10 extension matrix."""

    validate_extension_output_root(
        output_root,
        repository_root=repository_root,
        raw_data_roots=raw_data_roots,
        primary_evidence_roots=(
            *primary_evidence_roots,
            Path(repository_root) / "results/canonical",
        ),
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_capacity_output_layout(output_root)
    _assert_no_external_artifacts(output_root)
    if training.seeds != tuple(range(33, 43)):
        raise CapacityDataRegimePipelineError("extension training must use seeds 33 through 42")
    matrix = extension_run_matrix(protocol)
    if tuple(cohorts) != tuple(protocol.training_sizes):
        raise CapacityDataRegimePipelineError("cohort sizes do not match extension protocol")
    validated_core_bundle = build_lock_core(core_bundle["lock_core"])
    if validated_core_bundle["lock_core_sha256"] != core_bundle.get("lock_core_sha256"):
        raise CapacityDataRegimePipelineError("extension lock core digest is invalid")
    core_bundle = validated_core_bundle
    core_payload = core_bundle["lock_core"]
    core_digest = core_bundle["lock_core_sha256"]
    expected_preflight = core_payload["preflight"]
    if dict(preflight_report) != expected_preflight:
        raise CapacityDataRegimePipelineError(
            "measured preflight report does not match the sealed lock core"
        )
    primary_snapshot = snapshot_primary_artifacts(
        primary_study_lock=primary_study_lock,
        primary_prediction_inventory=primary_prediction_inventory,
    )
    lifecycle_dir = output_root / "lifecycle"
    if not (lifecycle_dir / "lifecycle-draft.json").exists():
        create_lifecycle_sidecar(
            lifecycle_dir, state="draft", lock_sha256=core_digest
        )

    store = representation_store
    if store is None:
        store = ValidatedRepresentationStore.build(
            records=records,
            cache_root=cache_root,
            available_memory_bytes=available_memory_bytes,
        )
    if getattr(store, "subject_count", None) != len(records):
        raise CapacityDataRegimePipelineError("validated representation store is incomplete")
    observed_window_count = sum(
        int(store.tensor(record.subject_id, -1).shape[0]) for record in records
    )
    if observed_window_count != preflight_report["cached_window_count"]:
        raise CapacityDataRegimePipelineError(
            "cached window count differs from the sealed preflight report"
        )

    summary_stores = {
        "mean": build_capacity_summary_store(
            records=records,
            representation_store=store,
            head_name="mean_linear",
            compute_device=device,
        ),
        "rich_stats": build_capacity_summary_store(
            records=records,
            representation_store=store,
            head_name="mean_rich_stats_residual",
            compute_device=device,
        ),
    }

    run_callable = train_frozen_probe_run if run_callable is None else run_callable
    run_records: list[dict[str, Any]] = []
    completed = 0
    for training_size in protocol.training_sizes:
        selected_records = select_capacity_records(
            records, tuple(cohorts[training_size])
        )
        train_records = tuple(record for record in selected_records if record.split == "train")
        flat_stores: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
        for summary_mode, summary_store in summary_stores.items():
            tensors = {
                record.subject_id: summary_store.tensor(record.subject_id, -1)
                for record in train_records
            }
            batch_plan = GlobalWindowBatchPlan.build(
                train_records, tensors=tensors, batch_size=training.batch_size
            )
            cpu_features, cpu_targets = batch_plan.flatten(tensors)
            flat_stores[summary_mode] = _prepare_training_store_device(
                cpu_features,
                cpu_targets,
                device=device,
            )
        for identity in (
            item for item in matrix if item.training_size == training_size
        ):
            summary_mode = capacity_summary_mode(identity.head)
            summary_store = summary_stores[summary_mode]
            flat_store = flat_stores[summary_mode]
            run_dir = (
                output_root
                / f"n-{training_size}"
                / _head_directory_name(identity.head)
                / f"seed-{identity.seed}"
            )
            run_context = {
                "extension_id": protocol.extension_id,
                "training_size": training_size,
                "head": identity.head,
                "hidden_dim": 4 if "hidden_dim=4" in identity.head else None,
                "head_complexity": capacity_head_complexity_metadata(
                    identity.head,
                    embed_dim=summary_store.embed_dim,
                    n_outputs=1,
                ),
                "feature_transform": {
                    "name": "per_window_token_summary",
                    "summary_mode": summary_mode,
                    "embed_dim": summary_store.embed_dim,
                },
                "resource": {
                    "requested_device": device,
                    "training_store_device": flat_store[2],
                    "training_batch_device": device,
                    "training_transfer_strategy": (
                        "chunked_cpu_to_cuda"
                        if flat_store[2] == "cpu" and device.startswith("cuda")
                        else "resident_store"
                    ),
                    "gpu_stage_chunk_windows": DEFAULT_GPU_STAGE_CHUNK_WINDOWS,
                    "inference_device": device,
                },
            }
            result = run_callable(
                head_name=identity.head,
                seed=identity.seed,
                records=selected_records,
                cache_root=cache_root,
                run_dir=run_dir,
                training=training,
                device=device,
                training_source_sha256=training_source_sha256,
                representation_store=summary_store,
                flat_training_store=flat_store,
                head_builder=build_capacity_data_regime_summary_head,
                required_layer_resolver=required_capacity_layer_for_head,
                run_metadata=run_context,
                progress_sink=progress_sink,
            )
            run_records.append(_capacity_run_record(identity, result.manifest))
            completed += 1
            if progress_sink is not None:
                progress_sink(
                    {
                        "event": "capacity_run_progress",
                        "completed_runs": completed,
                        "total_runs": 90,
                        "training_size": training_size,
                        "head": identity.head,
                        "seed": identity.seed,
                        "reused": result.reused,
                    }
                )

    _assert_primary_snapshot_unchanged(
        before=primary_snapshot,
        primary_study_lock=primary_study_lock,
        primary_prediction_inventory=primary_prediction_inventory,
    )
    _assert_no_external_artifacts(output_root)
    checkpoint_inventory = build_checkpoint_inventory(
        core_bundle,
        runs=run_records,
        expected_prediction_count=protocol.expected_prediction_count,
    )
    _write_json_exact(output_root / "checkpoint_inventory.json", checkpoint_inventory)
    checkpoint_lock = build_checkpoint_sealed_lock(core_bundle, checkpoint_inventory)
    _write_json_exact(output_root / "checkpoint_sealed_lock.json", checkpoint_lock)
    if not (lifecycle_dir / "lifecycle-checkpoint_sealed.json").exists():
        transition_lifecycle(
            lifecycle_dir,
            state="checkpoint_sealed",
            lock_sha256=core_digest,
        )
    return {
        "status": "checkpoint_sealed",
        "run_count": len(run_records),
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_sealed_lock": checkpoint_lock,
    }


def run_capacity_pilot(
    *,
    protocol: CapacityDataRegimeProtocol,
    training: FrozenProbeTrainingProtocol,
    records: Sequence[CachedSubjectRecord],
    cohorts: Mapping[int, Sequence[str]],
    cache_root: Path,
    output_root: Path,
    device: str,
    training_source_sha256: str,
    available_memory_bytes: int | None = None,
    representation_store: Any | None = None,
    run_callable: Callable[..., FrozenProbeRunResult] | None = None,
) -> Mapping[str, Any]:
    """Run the one preflight pilot used to measure extension wall time.

    The pilot deliberately uses a real extension identity (``n=200``,
    ``mean_linear``, seed ``33``) and the same cache/store/training hooks as
    the full matrix.  Its output lives in a separate sibling directory and
    is never admitted to the checkpoint inventory.
    """

    validate_extension_output_root(
        output_root,
        repository_root=Path.cwd(),
        raw_data_roots=(),
        primary_evidence_roots=(),
    )
    if training.seeds != tuple(range(33, 43)):
        raise CapacityDataRegimePipelineError("pilot training must use seeds 33 through 42")
    if 200 not in cohorts:
        raise CapacityDataRegimePipelineError("pilot requires the n=200 cohort")
    selected_records = select_capacity_records(records, tuple(cohorts[200]))
    store = representation_store
    if store is None:
        store = ValidatedRepresentationStore.build(
            records=records,
            cache_root=cache_root,
            available_memory_bytes=available_memory_bytes,
        )
    summary_store = build_capacity_summary_store(
        records=records,
        representation_store=store,
        head_name="mean_linear",
        compute_device=device,
    )
    train_records = tuple(record for record in selected_records if record.split == "train")
    tensors = {
        record.subject_id: summary_store.tensor(record.subject_id, -1)
        for record in train_records
    }
    batch_plan = GlobalWindowBatchPlan.build(
        train_records, tensors=tensors, batch_size=training.batch_size
    )
    cpu_features, cpu_targets = batch_plan.flatten(tensors)
    flat_store = _prepare_training_store_device(cpu_features, cpu_targets, device=device)
    run_context = {
        "extension_id": protocol.extension_id,
        "training_size": 200,
        "head": "mean_linear",
        "hidden_dim": None,
        "head_complexity": capacity_head_complexity_metadata(
            "mean_linear", embed_dim=summary_store.embed_dim, n_outputs=1
        ),
        "feature_transform": {
            "name": "per_window_token_summary",
            "summary_mode": "mean",
            "embed_dim": summary_store.embed_dim,
        },
        "resource": {
            "requested_device": device,
            "training_store_device": flat_store[2],
            "training_batch_device": device,
            "training_transfer_strategy": (
                "chunked_cpu_to_cuda"
                if flat_store[2] == "cpu" and device.startswith("cuda")
                else "resident_store"
            ),
            "gpu_stage_chunk_windows": DEFAULT_GPU_STAGE_CHUNK_WINDOWS,
            "inference_device": device,
        },
        "pilot": True,
    }
    run_callable = train_frozen_probe_run if run_callable is None else run_callable
    result = run_callable(
        head_name="mean_linear",
        seed=33,
        records=selected_records,
        cache_root=cache_root,
        run_dir=Path(output_root) / "n-200" / "mean_linear" / "seed-33",
        training=training,
        device=device,
        training_source_sha256=training_source_sha256,
        representation_store=summary_store,
        flat_training_store=flat_store,
        head_builder=build_capacity_data_regime_summary_head,
        required_layer_resolver=required_capacity_layer_for_head,
        run_metadata=run_context,
    )
    runtime = result.manifest.get("runtime_seconds")
    if not isinstance(runtime, (int, float)) or runtime <= 0 or not math.isfinite(float(runtime)):
        raise CapacityDataRegimePipelineError(
            "pilot manifest must contain a positive finite runtime_seconds"
        )
    return {
        "status": "pilot_complete",
        "training_size": 200,
        "head": "mean_linear",
        "seed": 33,
        "observed_pilot_seconds": float(runtime),
        "run_manifest_sha256": result.manifest.get("run_manifest_sha256"),
        "output_root": str(Path(output_root).resolve()),
        "reused": result.reused,
    }


def _validate_capacity_head_name(head_name: str) -> None:
    if head_name not in CAPACITY_HEADS:
        raise ValueError(
            "head must be in the exact extension allowlist: "
            + ", ".join(CAPACITY_HEADS)
        )


def _capacity_summary_vector(summary: Any, *, width: int) -> torch.Tensor:
    if (
        not isinstance(summary, torch.Tensor)
        or summary.ndim != 3
        or summary.shape[1] != 1
        or summary.shape[2] != width
    ):
        raise CapacityDataRegimePipelineError(
            "capacity summary head requires [batch, 1, summary_width] input"
        )
    return summary[:, 0, :]


class _CapacityMeanSummaryLinearHead(nn.Module):
    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, n_outputs)

    def forward(self, summary: Any) -> torch.Tensor:
        mean = _capacity_summary_vector(summary, width=self.embed_dim)
        return self.linear(mean)


class _CapacityMeanSummaryMLPResidualHead(nn.Module):
    def __init__(self, *, embed_dim: int, n_outputs: int, hidden_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.correction = nn.Linear(hidden_dim, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, summary: Any) -> torch.Tensor:
        mean = _capacity_summary_vector(summary, width=self.embed_dim)
        nonlinear_features = torch.nn.functional.gelu(self.hidden(mean))
        return self.linear(mean) + self.correction(nonlinear_features)


class _CapacityRichStatsSummaryHead(nn.Module):
    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.correction_scale = 0.5
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.correction = nn.Linear(4 * embed_dim, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, summary: Any) -> torch.Tensor:
        values = _capacity_summary_vector(summary, width=5 * self.embed_dim)
        mean = values[:, : self.embed_dim]
        stats = values[:, self.embed_dim :]
        return self.linear(mean) + self.correction_scale * self.correction(stats)


def build_capacity_data_regime_head(
    head_name: str, *, embed_dim: int, n_outputs: int = 1
) -> nn.Module:
    """Build an extension head without registering it as a primary head."""

    _validate_capacity_head_name(head_name)
    if head_name == "mean_linear":
        return MeanLinearCopyHead(embed_dim=embed_dim, n_outputs=n_outputs)
    if head_name == "mean_rich_stats_residual":
        return MeanRichStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
    return MeanMLPResidualHead(
        embed_dim=embed_dim,
        n_outputs=n_outputs,
        hidden_dim=4,
    )


def build_capacity_data_regime_summary_head(
    head_name: str, *, embed_dim: int, n_outputs: int = 1
) -> nn.Module:
    """Build a capacity head that consumes a fixed per-window summary."""

    _validate_capacity_head_name(head_name)
    if head_name == "mean_linear":
        return _CapacityMeanSummaryLinearHead(
            embed_dim=embed_dim,
            n_outputs=n_outputs,
        )
    if head_name == "mean_rich_stats_residual":
        return _CapacityRichStatsSummaryHead(
            embed_dim=embed_dim,
            n_outputs=n_outputs,
        )
    return _CapacityMeanSummaryMLPResidualHead(
        embed_dim=embed_dim,
        n_outputs=n_outputs,
        hidden_dim=4,
    )


def capacity_head_complexity_metadata(
    head_name: str, *, embed_dim: int, n_outputs: int = 1
) -> dict[str, Any]:
    """Return complexity metadata with the extension identity preserved."""

    _validate_capacity_head_name(head_name)
    if head_name == "mean_mlp_residual_matched(hidden_dim=4)":
        metadata = head_complexity_metadata(
            "mean_mlp_residual",
            embed_dim=embed_dim,
            n_outputs=n_outputs,
            hidden_dim=4,
        )
        return {
            **metadata,
            "head_name": head_name,
            "hidden_dim": 4,
        }
    variant = head_name
    metadata = head_complexity_metadata(
        variant, embed_dim=embed_dim, n_outputs=n_outputs
    )
    return {**metadata, "head_name": head_name, "hidden_dim": None}
