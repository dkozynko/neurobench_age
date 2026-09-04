"""Schema-versioned evidence primitives for experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


SCHEMA_VERSION = "1.0"
_STATUSES = {"running", "completed", "failed", "partial"}
_EVALUATION_MODES = {"validation_only", "final_test"}
_DETERMINISTIC_POLICIES = {"strict", "best_effort"}
_SOURCE_SUFFIXES = {
    ".bash",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_EXCLUDED_SOURCE_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "results",
}
_DEFAULT_COMPARISON_FACTOR_KEYS = frozenset(
    {
        "head_variant",
        "head_complexity",
        "head_dropout",
        "layer_index",
        "layer_indices",
        "layer_mix_alpha",
        "mean_gradient_scale",
        "correction_gradient_scale",
        "swa_window",
        "correlation_loss_lambda",
        "correlation_loss_objective",
        "robust_loss",
        "target_scaler_mode",
        "two_stage_finetune",
        "two_stage_warmup_epochs",
        "two_stage_unfreeze_last_blocks",
        "two_stage_encoder_gradient_scale",
        "augmentation_consistency",
        "augmentation_consistency_lambda",
        "augmentation_noise_scale",
        "augmentation_consistency_batch_size",
        "augmentation_space",
        "augmentation_scope",
        "augmentation_pairing",
        "continued_pretraining",
        "pretraining_epochs",
        "pretraining_mask_fraction",
        "pretraining_mask_block_samples",
        "pretraining_learning_rate",
        "pretraining_weight_decay",
        "pretraining_max_batches",
        "pretraining_source_split",
        "pretraining_objective",
        "pretraining_age_labels_used",
        "H7_HEAD_VARIANT",
        "H7_LAYER_INDEX",
        "H7_LAYER_INDICES",
        "H7_LAYER_MIX_ALPHA",
        "H7_MEAN_GRADIENT_SCALE",
        "H7_CORRECTION_GRADIENT_SCALE",
        "SWA_WINDOW",
        "CORRELATION_LOSS_LAMBDA",
        "CORRELATION_LOSS_OBJECTIVE",
        "ROBUST_LOSS",
        "TARGET_SCALER_MODE",
        "TWO_STAGE_FINETUNE",
        "TWO_STAGE_WARMUP_EPOCHS",
        "TWO_STAGE_UNFREEZE_LAST_BLOCKS",
        "TWO_STAGE_ENCODER_GRADIENT_SCALE",
        "AUGMENTATION_CONSISTENCY",
        "AUGMENTATION_CONSISTENCY_LAMBDA",
        "AUGMENTATION_NOISE_SCALE",
        "AUGMENTATION_CONSISTENCY_BATCH_SIZE",
        "AUGMENTATION_SPACE",
        "AUGMENTATION_SCOPE",
        "CONTINUED_PRETRAINING",
        "PRETRAINING_EPOCHS",
        "PRETRAINING_MASK_FRACTION",
        "PRETRAINING_MASK_BLOCK_SAMPLES",
        "PRETRAINING_LEARNING_RATE",
        "PRETRAINING_WEIGHT_DECAY",
        "PRETRAINING_MAX_BATCHES",
        "PRETRAINING_SOURCE_SPLIT",
        "PRETRAINING_OBJECTIVE",
        "PRETRAINING_AGE_LABELS_USED",
    }
)
_RUNTIME_ONLY_CONFIG_KEYS = frozenset(
    {
        "acquisition_provenance_path",
        "command_line",
        "gpu_hourly_rate_usd",
        "launch_command",
        "output_dir",
        "provenance_path",
        "run_dir",
        "CACHE_DIR",
        "DATA_DIR",
        "SAVE_DIR",
    }
)


class ResourceProbe(Protocol):
    def reset_peak_memory_stats(self) -> None: ...

    def snapshot(self) -> Mapping[str, Any]: ...


class _DefaultResourceProbe:
    def reset_peak_memory_stats(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            return

    def snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "gpu_model": None,
            "gpu_count": 0,
            "gpu_vram_mb": None,
            "hardware_class": None,
            "cuda": None,
            "driver": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
            "peak_cpu_rss_mb": None,
        }
        try:
            import torch

            snapshot["cuda"] = getattr(torch.version, "cuda", None)
            if torch.cuda.is_available():
                device_index = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device_index)
                model = torch.cuda.get_device_name(device_index)
                vram_mb = float(properties.total_memory) / (1024 * 1024)
                snapshot.update(
                    {
                        "gpu_model": model,
                        "gpu_count": torch.cuda.device_count(),
                        "gpu_vram_mb": vram_mb,
                        "hardware_class": f"{model}/{round(vram_mb)}MB",
                        "peak_allocated_mb": float(torch.cuda.max_memory_allocated()) / (1024 * 1024),
                        "peak_reserved_mb": float(torch.cuda.max_memory_reserved()) / (1024 * 1024),
                    }
                )
        except Exception:
            pass
        try:
            import psutil

            snapshot["peak_cpu_rss_mb"] = float(psutil.Process().memory_info().rss) / (1024 * 1024)
        except Exception:
            pass
        return snapshot


class _PhaseRecorder:
    def __init__(self, recorder: "EvidenceRecorder", name: str) -> None:
        self.recorder = recorder
        self.name = name
        self.started = 0.0
        self.batches = 0
        self.samples = 0

    def __enter__(self) -> "_PhaseRecorder":
        self.started = time.perf_counter()
        return self

    def record(self, *, batches: int = 0, samples: int = 0) -> None:
        if isinstance(batches, bool) or not isinstance(batches, int) or batches < 0:
            raise ValueError("phase batches must be a non-negative integer")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
            raise ValueError("phase samples must be a non-negative integer")
        self.batches += batches
        self.samples += samples

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        elapsed = max(0.0, time.perf_counter() - self.started)
        self.recorder._record_phase(
            self.name,
            elapsed_seconds=elapsed,
            batches=self.batches,
            samples=self.samples,
        )
        return False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode(
        "utf-8"
    )


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def source_tree_sha256(source_root: Path) -> str:
    """Hash reproducible source/config files while excluding runtime outputs."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest = hashlib.sha256()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_SOURCE_DIRECTORIES for part in relative.parts):
            continue
        files.append(path)
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _comparison_factor_keys(resolved_config: Mapping[str, Any]) -> frozenset[str]:
    declared = resolved_config.get("comparison_factor_keys")
    if declared is None:
        return _DEFAULT_COMPARISON_FACTOR_KEYS
    if not isinstance(declared, (list, tuple, set, frozenset)):
        raise ValueError("comparison_factor_keys must be a sequence of strings")
    if any(not isinstance(key, str) or not key for key in declared):
        raise ValueError("comparison_factor_keys must contain non-empty strings")
    return frozenset(declared)


def _remove_config_keys(value: Any, keys: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _remove_config_keys(item, keys)
            for key, item in value.items()
            if key not in keys
        }
    if isinstance(value, list):
        return [_remove_config_keys(item, keys) for item in value]
    if isinstance(value, tuple):
        return [_remove_config_keys(item, keys) for item in value]
    return value


def comparison_config_hash(resolved_config: Mapping[str, Any]) -> str:
    """Hash config fields that must remain equal in matched comparisons."""

    factor_keys = _comparison_factor_keys(resolved_config)
    return sha256_json(
        _remove_config_keys(
            resolved_config,
            factor_keys | _RUNTIME_ONLY_CONFIG_KEYS | {"comparison_factor_keys"},
        )
    )


def deterministic_policy_status(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return whether settings satisfy the strict reproducibility contract."""

    required = {
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    violations = [
        f"{key}={settings.get(key)!r}, expected={expected!r}"
        for key, expected in required.items()
        if settings.get(key) != expected
    ]
    return {"satisfied": not violations, "violations": violations}


def _repository_root() -> Path:
    """Resolve the checkout root when running from the ``src`` layout.

    Editable installs keep ``__file__`` inside the checkout, while a regular
    wheel install has no Git metadata. In the latter case the package parent is
    the best available source root and Git probing safely returns no revision.
    """

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return module_path.parents[2]


def _git_metadata() -> dict[str, Any]:
    """Capture repository state without making a run depend on Git."""

    repo_root = _repository_root()
    try:
        source_digest = source_tree_sha256(repo_root)
    except (OSError, ValueError):
        source_digest = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "source_tree_sha256": source_digest,
        }
    return {
        "commit": commit or None,
        "branch": branch or None,
        "dirty": dirty,
        "source_tree_sha256": source_digest,
    }


def _software_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": None,
        "lightning": None,
    }
    try:
        import torch

        metadata["torch"] = getattr(torch, "__version__", None)
        metadata["deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
    except Exception:
        pass
    try:
        import lightning

        metadata["lightning"] = getattr(lightning, "__version__", None)
    except Exception:
        try:
            import lightning.pytorch as lightning_pytorch

            metadata["lightning"] = getattr(lightning_pytorch, "__version__", None)
        except Exception:
            pass
    return metadata


def _deterministic_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        import torch

        metadata["torch_deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        metadata["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        metadata["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        metadata["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        metadata["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        metadata["torch_settings"] = "unavailable"
    return metadata


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write JSONL deterministically and return its SHA-256 digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(canonical_json_bytes(row) for row in rows)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def _counts(value: Mapping[str, Any], *, label: str) -> tuple[int, int, int]:
    try:
        total = value["total"]
        trainable = value["trainable"]
        frozen = value["frozen"]
    except KeyError as error:
        raise ValueError(f"{label} parameter bucket is missing {error.args[0]}") from error
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (total, trainable, frozen)):
        raise ValueError(f"{label} parameter counts must be integers")
    if min(total, trainable, frozen) < 0:
        raise ValueError(f"{label} parameter counts must be non-negative")
    if total != trainable + frozen:
        raise ValueError(f"{label} parameter total does not equal trainable plus frozen")
    return total, trainable, frozen


def validate_parameter_buckets(buckets: Mapping[str, Any]) -> None:
    encoder = _counts(buckets["encoder"], label="encoder")
    head = _counts(buckets["head"], label="head")
    auxiliary = buckets.get("auxiliary", [])
    if not isinstance(auxiliary, list):
        raise ValueError("auxiliary parameter buckets must be a list")
    auxiliary_totals = [
        _counts(item, label=f"auxiliary[{index}]") for index, item in enumerate(auxiliary)
    ]
    expected = tuple(
        left + right + sum(item[position] for item in auxiliary_totals)
        for position, (left, right) in enumerate(zip(encoder, head))
    )
    actual = _counts(buckets["total"], label="total")
    if actual != expected:
        raise ValueError(f"parameter total does not match buckets: expected {expected}, got {actual}")


def add_declared_head_bucket(
    buckets: Mapping[str, Any],
    *,
    parameter_count: int,
    trainable_parameter_count: int | None = None,
) -> dict[str, Any]:
    """Add an externally wrapped head to model-level parameter accounting.

    Some NeuralBench versions expose the encoder as ``BrainModule.model``
    while the downstream head is configured separately.  In that shape a
    model-only count reports a zero-sized head even though the run metadata
    has an authoritative head complexity contract.  This helper completes
    that accounting without changing already-correct measured buckets.
    """

    validate_parameter_buckets(buckets)
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count < 0:
        raise ValueError("declared head parameter_count must be a non-negative integer")
    if trainable_parameter_count is None:
        trainable_parameter_count = parameter_count
    if (
        isinstance(trainable_parameter_count, bool)
        or not isinstance(trainable_parameter_count, int)
        or trainable_parameter_count < 0
        or trainable_parameter_count > parameter_count
    ):
        raise ValueError("declared head trainable parameter count is invalid")

    current_head = _counts(buckets["head"], label="head")
    if current_head[0] not in {0, parameter_count}:
        raise ValueError(
            "measured head parameter count does not match declared head complexity: "
            f"measured={current_head[0]} declared={parameter_count}"
        )
    if current_head[0] == parameter_count:
        return {
            "encoder": dict(buckets["encoder"]),
            "head": dict(buckets["head"]),
            "auxiliary": [dict(item) for item in buckets.get("auxiliary", [])],
            "total": dict(buckets["total"]),
        }

    encoder = dict(buckets["encoder"])
    auxiliary = [dict(item) for item in buckets.get("auxiliary", [])]
    head = {
        "total": parameter_count,
        "trainable": trainable_parameter_count,
        "frozen": parameter_count - trainable_parameter_count,
    }
    auxiliary_totals = [
        _counts(item, label=f"auxiliary[{index}]") for index, item in enumerate(auxiliary)
    ]
    encoder_counts = _counts(encoder, label="encoder")
    total = {
        "total": encoder_counts[0] + head["total"] + sum(item[0] for item in auxiliary_totals),
        "trainable": encoder_counts[1] + head["trainable"] + sum(item[1] for item in auxiliary_totals),
        "frozen": encoder_counts[2] + head["frozen"] + sum(item[2] for item in auxiliary_totals),
    }
    completed = {"encoder": encoder, "head": head, "auxiliary": auxiliary, "total": total}
    validate_parameter_buckets(completed)
    return completed


def estimate_head_parameter_count(variant: str, *, embed_dim: int, n_outputs: int) -> int:
    try:
        from ..heads.math import head_complexity_metadata
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        # Keep the evidence primitives usable in a lightweight CPU/document
        # environment.  Runtime experiments use the authoritative contract in
        # reve_head_math; this fallback mirrors its parameter-count formulas.
        d = int(embed_dim)
        o = int(n_outputs)
        linear = (d + 1) * o
        if variant in {
            "mean_linear", "mean_linear_copy", "mean_linear_detached",
            "mean_linear_gradient_scaled", "mean_linear_probe_scaled",
            "mean_layer_linear", "mean_layer_mix_fixed",
        }:
            return linear
        if variant == "mean_layer_mix":
            return linear + 1
        if variant == "mean_linear_warmup":
            return 2 * linear + 1
        if variant == "mean_anchor":
            return d + 1 + linear
        if variant == "mean_residual":
            return d + linear + d * o
        if variant == "mean_vector_anchor":
            return 2 * d + linear
        if variant == "mean_stats_attention_residual":
            return d + linear + 3 * d * o
        if variant == "mean_attention_gated":
            return d + linear + d * o + 1
        if variant in {
            "mean_stats_residual", "mean_stats_residual_detached",
            "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled",
        }:
            return linear + 2 * d * o
        if variant == "global_stats_residual":
            return linear + 4 * o
        if variant in {"mean_rich_stats_residual", "mean_rich_stats_gradient_routes"}:
            return linear + 4 * d * o
        if variant == "mean_anchor_ensemble":
            return 2 * linear + 4 * d * o + 1
        if variant in {"mean_reliability_shrinkage", "mean_reliability_stable"}:
            return linear + 4 * d * o + 4
        if variant == "grouped_rich_stats_shrinkage":
            return linear + 4 * (d * d + d) + 4
        if variant == "grouped_stats_shared_gate":
            return linear + 4 * (d * d + d) + 1
        if variant == "temporal_pyramid_stats":
            return linear + (4 * 2 * d) * 8 + 8 * d
        if variant == "mean_covariance_residual":
            return linear + d * 4 + 4 * d
        if variant == "multi_query_rich_stats":
            return 2 * d + linear + 5 * 2 * d * o
        if variant in {"last_avg", "last"}:
            return 2 * d + linear
        if variant == "last_tuned":
            return 2 * d + linear + 1
        if variant == "all":
            return 2 * d + linear
        raise ValueError(f"unsupported head variant {variant!r}")
    metadata = head_complexity_metadata(variant, embed_dim=embed_dim, n_outputs=n_outputs)
    count = metadata.get("parameter_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"head parameter count is not defined for {variant!r}")
    return count


def count_parameters(module: Any) -> dict[str, int]:
    """Count total, trainable, and frozen parameters in a torch module."""

    parameters = list(module.parameters()) if callable(getattr(module, "parameters", None)) else []
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(int(parameter.numel()) for parameter in parameters if bool(getattr(parameter, "requires_grad", False)))
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def parameter_buckets_from_model(model: Any, *, head: Any | None = None, auxiliary: Iterable[tuple[str, Any]] = ()) -> dict[str, Any]:
    """Build the paper-facing encoder/head/auxiliary parameter accounting.

    ``head`` may be supplied explicitly.  Otherwise a model's ``head``
    attribute is used when present; all remaining parameters are assigned to
    the encoder bucket unless named auxiliary modules are supplied.
    """

    auxiliary = list(auxiliary)
    all_parameters = list(model.parameters()) if callable(getattr(model, "parameters", None)) else []
    head = head if head is not None else getattr(model, "head", None)
    head_parameters = list(head.parameters()) if head is not None and callable(getattr(head, "parameters", None)) else []
    auxiliary_rows = [{"name": name, **count_parameters(module)} for name, module in auxiliary]
    excluded = {id(parameter) for parameter in head_parameters}
    auxiliary_parameter_ids: set[int] = set()
    for _, module in auxiliary:
        if callable(getattr(module, "parameters", None)):
            auxiliary_parameter_ids.update(id(parameter) for parameter in module.parameters())
    encoder_parameters = [
        parameter for parameter in all_parameters
        if id(parameter) not in excluded and id(parameter) not in auxiliary_parameter_ids
    ]
    def _counts(parameters: Iterable[Any]) -> dict[str, int]:
        values = list(parameters)
        total = sum(int(parameter.numel()) for parameter in values)
        trainable = sum(int(parameter.numel()) for parameter in values if bool(getattr(parameter, "requires_grad", False)))
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
    encoder = _counts(encoder_parameters)
    head_counts = _counts(head_parameters)
    total = _counts(all_parameters)
    validate_parameter_buckets({"encoder": encoder, "head": head_counts, "auxiliary": auxiliary_rows, "total": total})
    return {"encoder": encoder, "head": head_counts, "auxiliary": auxiliary_rows, "total": total}


class EvidenceRecorder:
    """Write a run manifest and collect evidence without changing training."""

    def __init__(
        self,
        run_dir: Path,
        *,
        run_id: str,
        task: str,
        dataset_manifest: str,
        split_fingerprint: str,
        seed: int,
        resolved_config: Mapping[str, Any],
        command_line: str,
        evaluation_mode: str = "validation_only",
        deterministic_policy: str = "best_effort",
        resource_probe: ResourceProbe | None = None,
        gpu_hourly_rate_usd: float | None = None,
    ) -> None:
        if evaluation_mode not in _EVALUATION_MODES:
            raise ValueError(f"unsupported evaluation mode: {evaluation_mode}")
        if deterministic_policy not in _DETERMINISTIC_POLICIES:
            raise ValueError(f"unsupported deterministic policy: {deterministic_policy}")
        comparison_factor_keys = _comparison_factor_keys(resolved_config)
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.evaluation_mode = evaluation_mode
        self.resource_probe = resource_probe or _DefaultResourceProbe()
        self.gpu_hourly_rate_usd = gpu_hourly_rate_usd
        self._phases: dict[str, dict[str, Any]] = {}
        self._parameter_buckets: dict[str, Any] = {}
        self._head_complexity: dict[str, Any] = {}
        self._optimizer: dict[str, Any] = {}
        self._throughput: dict[str, Any] = {}
        git_metadata = _git_metadata()
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "created",
            "task": task,
            "dataset_manifest": dataset_manifest,
            "split_fingerprint": split_fingerprint,
            "seed": seed,
            "config_hash": sha256_json(resolved_config),
            "protocol_digest": sha256_json(resolved_config.get("protocol", {})),
            "comparison_config_hash": comparison_config_hash(resolved_config),
            "comparison_factor_keys": sorted(comparison_factor_keys),
            "git": git_metadata,
            "source_tree_sha256": git_metadata.get("source_tree_sha256"),
            "hardware": {
                "host": socket.gethostname(),
                "gpu_model": None,
                "gpu_count": 0,
                "gpu_vram_mb": None,
                "hardware_class": None,
                "cuda": None,
                "driver": None,
            },
            "software": _software_metadata(),
            "resolved_config": dict(resolved_config),
            "command_line": command_line,
            "deterministic_settings": {},
            "deterministic_policy": deterministic_policy,
            "deterministic_policy_satisfied": None,
            "deterministic_policy_violations": [],
            "precision": (
                resolved_config.get("protocol", {}).get("precision")
                if isinstance(resolved_config.get("protocol"), Mapping)
                else resolved_config.get("precision")
            ),
            "evaluation_mode": evaluation_mode,
            "test_access": (
                "single_use_predeclared"
                if evaluation_mode == "final_test"
                else "sealed"
            ),
            "analysis_spec_hash": None,
            "train_age_reference_path": None,
            "train_age_reference_sha256": None,
            "missing": [],
            "started_at_utc": None,
            "ended_at_utc": None,
            "failure_reason": None,
        }
        self._monotonic_start: float | None = None

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    def start(self) -> dict[str, Any]:
        if self._manifest["status"] != "created":
            raise RuntimeError("evidence recorder can only be started once")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest["status"] = "running"
        self._manifest["started_at_utc"] = utc_now_iso()
        self._monotonic_start = time.perf_counter()
        self.resource_probe.reset_peak_memory_stats()
        self._manifest["deterministic_settings"] = _deterministic_metadata()
        policy_status = deterministic_policy_status(self._manifest["deterministic_settings"])
        self._manifest["deterministic_policy_satisfied"] = (
            True
            if self._manifest["deterministic_policy"] == "best_effort"
            else policy_status["satisfied"]
        )
        self._manifest["deterministic_policy_violations"] = policy_status["violations"]
        self._update_hardware(self.resource_probe.snapshot())
        self._write_manifest()
        return dict(self._manifest)

    def set_parameter_buckets(self, buckets: Mapping[str, Any]) -> None:
        validate_parameter_buckets(buckets)
        self._parameter_buckets = dict(buckets)

    def set_head_complexity(self, metadata: Mapping[str, Any]) -> None:
        required = ("variant", "input_width", "output_width", "operations", "parameter_count_formula")
        missing = [field for field in required if field not in metadata]
        if missing:
            raise ValueError(f"head complexity metadata is missing {missing}")
        self._head_complexity = dict(metadata)

    def set_optimizer_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._optimizer = dict(metadata)

    def set_throughput_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._throughput = dict(metadata)

    def set_manifest_metadata(self, **metadata: Any) -> None:
        """Set late-bound reproducibility fields before finalization."""

        allowed = {
            "git", "software", "deterministic_settings", "precision", "analysis_spec_hash",
            "test_access", "config_hash", "dataset_manifest", "split_fingerprint",
            "source_tree_sha256", "protocol_digest", "comparison_config_hash",
            "train_age_reference_path", "train_age_reference_sha256",
        }
        unknown = sorted(set(metadata) - allowed)
        if unknown:
            raise ValueError(f"unsupported manifest metadata fields: {unknown}")
        for key, value in metadata.items():
            if key in {"git", "software", "deterministic_settings"} and isinstance(value, Mapping):
                self._manifest[key].update(dict(value))
            else:
                self._manifest[key] = value
        if self._manifest["status"] != "created":
            self._write_manifest()

    def phase(self, name: str) -> _PhaseRecorder:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("phase name must be a non-empty string")
        return _PhaseRecorder(self, name)

    def _record_phase(self, name: str, *, elapsed_seconds: float, batches: int, samples: int) -> None:
        self._phases[name] = {
            "elapsed_seconds": float(elapsed_seconds),
            "batches": int(batches),
            "samples": int(samples),
            "batches_per_second": float(batches / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
            "samples_per_second": float(samples / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        }

    def _update_hardware(self, snapshot: Mapping[str, Any]) -> None:
        for key in (
            "gpu_model",
            "gpu_count",
            "gpu_vram_mb",
            "hardware_class",
            "cuda",
            "driver",
        ):
            value = snapshot.get(key)
            if value is not None:
                self._manifest["hardware"][key] = value

    def _complexity_payload(self) -> dict[str, Any]:
        snapshot = dict(self.resource_probe.snapshot())
        self._update_hardware(snapshot)
        missing = list(self._manifest["missing"])
        if snapshot.get("peak_allocated_mb") is None:
            reason = "GPU peak memory unavailable (CUDA not available)"
            if reason not in missing:
                missing.append(reason)
        elapsed_seconds = None
        if self._monotonic_start is not None:
            elapsed_seconds = max(0.0, time.perf_counter() - self._monotonic_start)
        cost = None
        if elapsed_seconds is not None and self.gpu_hourly_rate_usd is not None:
            cost = elapsed_seconds / 3600.0 * float(self.gpu_hourly_rate_usd)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self._manifest["status"],
            "parameter_buckets": self._parameter_buckets,
            "parameter_accounting_status": "complete" if self._parameter_buckets else "unavailable",
            "head_complexity": self._head_complexity,
            "optimizer": self._optimizer,
            "phases": self._phases,
            "throughput": self._throughput,
            "memory": {
                "peak_allocated_mb": snapshot.get("peak_allocated_mb"),
                "peak_reserved_mb": snapshot.get("peak_reserved_mb"),
                "peak_cpu_rss_mb": snapshot.get("peak_cpu_rss_mb"),
            },
            "hardware": dict(self._manifest["hardware"]),
            "cost_usd": cost,
            "missing": missing,
        }

    def add_missing(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("missing evidence reason must be a non-empty string")
        if reason not in self._manifest["missing"]:
            self._manifest["missing"].append(reason)
        if self._manifest["status"] != "created":
            self._write_manifest()

    def finalize(self, status: str, *, error: str | None = None) -> dict[str, Any]:
        if status not in _STATUSES - {"running"}:
            raise ValueError(f"invalid final evidence status: {status}")
        if self._manifest["status"] not in {"running", "created"}:
            raise RuntimeError("evidence recorder is already finalized")
        self._manifest["status"] = status
        self._manifest["ended_at_utc"] = utc_now_iso()
        if error is not None:
            self._manifest["failure_reason"] = str(error)
        complexity = self._complexity_payload()
        self._manifest["missing"] = list(complexity["missing"])
        self._write_manifest()
        write_json_atomic(self.run_dir / "complexity.json", complexity)
        return dict(self._manifest)

    def _write_manifest(self) -> None:
        write_json_atomic(self.manifest_path, self._manifest)
