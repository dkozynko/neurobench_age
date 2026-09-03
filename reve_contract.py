"""Stable protocol constants and validation helpers for REVE heads."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

UPSTREAM_REVE_REPOSITORY = "https://github.com/elouayas/reve_eeg"
UPSTREAM_REVE_COMMIT = "06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7"
UPSTREAM_REVE_FILE_HASHES = {
    "src/models/classifier.py": "cf71d0f455df0e9a263363213d75fbf493ffc5d587cd3c1f80cdc90ab961f90f",
    "src/models/encoder.py": "933684a3306e16c16926cead0fc8c8c0a9ae99ad9dfaf54dee860c5b335d1b6c",
    "src/models/backbone.py": "632ecd10ddb5efcc6f50a7207b0aa3934e7304438eb3f1db1edfefc3fab3371f",
}

OFFICIAL_HEAD_VARIANTS = ("mean_linear", "last_avg", "last", "all")
LOCAL_HEAD_VARIANTS = (
    "mean_linear_copy",
    "mean_linear_detached",
    "mean_linear_warmup",
    "mean_linear_gradient_scaled",
    "mean_linear_probe_scaled",
    "mean_anchor",
    "mean_residual",
    "mean_vector_anchor",
    "mean_mlp_residual",
    "mean_stats_residual",
    "mean_stats_residual_detached",
    "mean_stats_residual_gradient_scaled",
    "mean_stats_probe_scaled",
    "mean_stats_attention_residual",
    "mean_attention_gated",
    "global_stats_residual",
    "mean_rich_stats_residual",
    "mean_rich_stats_gradient_routes",
    "mean_anchor_ensemble",
    "mean_reliability_shrinkage",
    "mean_reliability_stable",
    "grouped_rich_stats_shrinkage",
    "grouped_stats_shared_gate",
    "temporal_pyramid_stats",
    "mean_covariance_residual",
    "multi_query_rich_stats",
    "mean_layer_linear",
    "mean_layer_mix",
    "mean_layer_mix_fixed",
)
LAST_TUNED_PROTOCOL_VARIANTS = ("last_tuned",)
HEAD_VARIANTS = OFFICIAL_HEAD_VARIANTS + LOCAL_HEAD_VARIANTS + LAST_TUNED_PROTOCOL_VARIANTS
UPSTREAM_HEAD_VARIANTS = ("last_avg", "last", "all")
LAST_TUNED_HEAD_SOURCE = "upstream_reve_tuned"
LAST_TUNED_HEAD_ARCHITECTURE = "last_tuned_residual_query_attention"
LAST_TUNED_PROTOCOL_CLASS = "tuning"
LAST_TUNED_INITIAL_ALPHA = 0.1
LAST_TUNED_BASE_LR = 1e-4
LAST_TUNED_QUERY_LR = 1e-5
LAST_TUNED_WEIGHT_DECAY = 0.05
LAST_TUNED_SCHEDULER_MAX_LR = (LAST_TUNED_BASE_LR, LAST_TUNED_QUERY_LR)
LAST_TUNED_SCHEDULER_PCT_START = 0.1
LAST_TUNED_SCHEDULER_DIV_FACTOR = 25.0
LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR = 1e4
DEFAULT_UPSTREAM_DROPOUT = 0.0
RMS_NORM_EPS = 1e-6
UPSTREAM_HEAD_HIDDEN_SIZE = 512
UPSTREAM_HEAD_INIT_STD = UPSTREAM_HEAD_HIDDEN_SIZE ** -0.5
UPSTREAM_HEAD_INIT_CUTOFF = 3.0
_NEURALBENCH_TRAIN_DUMMY_CONTEXT = object()

class AdapterContractError(ValueError):
    """Raised when an encoder output cannot be interpreted safely."""


class ProtocolMismatchError(ValueError):
    """Raised before training when the resolved official protocol differs."""


def validate_head_variant(variant: str) -> str:
    """Validate and return a supported head variant."""

    if variant not in HEAD_VARIANTS:
        raise ValueError(f"unsupported REVE head variant {variant!r}; expected one of {HEAD_VARIANTS}")
    return variant


def validate_official_head_variant(variant: str) -> str:
    """Validate a head variant admitted by the official protocol."""

    if variant not in OFFICIAL_HEAD_VARIANTS:
        raise ValueError(f"{variant!r} is not an official REVE head variant; expected one of {OFFICIAL_HEAD_VARIANTS}")
    return variant


def validate_local_head_variant(variant: str) -> str:
    """Validate a local control head that is not part of the official registry."""

    if variant not in LOCAL_HEAD_VARIANTS:
        raise ValueError(f"{variant!r} is not a local REVE head variant; expected one of {LOCAL_HEAD_VARIANTS}")
    return variant


def validate_last_tuned_protocol(
    variant: str,
    *,
    experiment: Any | None = None,
    optimizer_config: Mapping[str, Any] | None = None,
) -> str:
    """Validate the isolated ``last_tuned`` protocol.

    The one-argument form intentionally checks only the variant so pure head
    construction stays independent of NeuralBench.  Supplying either resolved
    runtime object opts into the complete fail-closed tuning contract.
    """

    if variant not in LAST_TUNED_PROTOCOL_VARIANTS:
        raise ValueError(f"{variant!r} is not a last_tuned protocol variant; expected one of {LAST_TUNED_PROTOCOL_VARIANTS}")
    if experiment is None and optimizer_config is None:
        return variant
    if experiment is None or optimizer_config is None:
        raise ProtocolMismatchError("last_tuned resolved validation requires both experiment and optimizer_config")
    _validate_last_tuned_resolved_protocol(experiment, optimizer_config)
    return variant


def validate_upstream_head_variant(variant: str) -> str:
    """Validate and return a non-baseline upstream head variant."""

    validate_official_head_variant(variant)
    if variant not in UPSTREAM_HEAD_VARIANTS:
        raise ValueError(f"{variant!r} is the official baseline, not an upstream head variant")
    return variant

PROTOCOL_CONTRACT: dict[str, Any] = {
    "max_epochs": 40,
    "batch_size": 64,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "scheduler": "OneCycleLR",
    "scheduler_max_lr": 1e-4,
    "scheduler_pct_start": 0.1,
    "scheduler_anneal_strategy": "cos",
    "gradient_clip_val": 1.0,
    "precision": "32-true",
    "loss": "MSELoss",
    "monitor": "val/pearsonr",
    "mode": "max",
    "patience": 7,
    "batch_num_workers": 2,
    "persistent_workers": True,
    "target_scaler": None,
    "frequency": 200.0,
    "filter": [0.5, 99.5],
    "notch_filter": None,
    "scaler": "StandardScaler",
    "clamp": 15,
    "window_duration_s": 2.0,
    "window_stride_s": 2.0,
}


# ---------------------------------------------------------------------------
# Protocol, source-lock, and runtime metadata
# ---------------------------------------------------------------------------


_MISSING = object()


def _get_path(value: Any, path: str, default: Any = _MISSING) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        else:
            owner = current
            current = getattr(owner, part, _MISSING)
            # NeuralTrain stores discriminator names in ``model_dump`` but
            # does not expose them as model attributes. This matters for
            # optimizer/scheduler/loss protocol checks such as ``name``.
            if current is _MISSING and hasattr(owner, "model_dump"):
                current = owner.model_dump().get(part, _MISSING)
        if current is _MISSING:
            return default
    return current


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(actual, tuple):
        actual = list(actual)
    return actual == expected


def _path_mismatches(
    value: Any,
    checks: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    """Return readable mismatch messages for a mapping/object path contract."""

    mismatches: list[str] = []
    for path, expected in checks.items():
        actual = _get_path(value, path)
        if actual is _MISSING or not _same(actual, expected):
            mismatches.append(f"{prefix}{path}: expected {expected!r}, got {actual!r}")
    return mismatches


def _last_tuned_group_mismatches(groups: Any) -> list[str]:
    """Validate the two named metadata groups used by ``last_tuned``."""

    if not isinstance(groups, (list, tuple)) or len(groups) != 2:
        return ["optimizer_config.param_groups must contain exactly base and query"]

    mismatches: list[str] = []
    expected_groups = (("base", LAST_TUNED_BASE_LR), ("query", LAST_TUNED_QUERY_LR))
    seen_parameter_names: set[str] = set()
    for index, (expected_name, expected_lr) in enumerate(expected_groups):
        group = groups[index]
        group_path = f"optimizer_config.param_groups[{index}]"
        if not isinstance(group, Mapping) and not hasattr(group, "__dict__"):
            mismatches.append(f"{group_path} must be a mapping-like object")
            continue

        expected_fields = {
            "name": expected_name,
            "learning_rate": expected_lr,
            "weight_decay": LAST_TUNED_WEIGHT_DECAY,
        }
        for field, expected in expected_fields.items():
            actual = _get_path(group, field)
            if actual is _MISSING or not _same(actual, expected):
                mismatches.append(f"{group_path}.{field}: expected {expected!r}, got {actual!r}")

        parameter_names = _get_path(group, "parameter_names")
        if (
            isinstance(parameter_names, str)
            or not isinstance(parameter_names, (list, tuple))
            or not all(isinstance(name, str) and name for name in parameter_names)
        ):
            mismatches.append(f"{group_path}.parameter_names: expected a non-empty sequence of names, got {parameter_names!r}")
            continue
        if len(parameter_names) != len(set(parameter_names)):
            mismatches.append(f"{group_path}.parameter_names: duplicate parameter names")
        overlap = seen_parameter_names.intersection(parameter_names)
        if overlap:
            mismatches.append(f"optimizer_config.param_groups.parameter_names: duplicate names {sorted(overlap)!r}")
        seen_parameter_names.update(parameter_names)

        if expected_name == "base":
            if "head.query_token" in parameter_names:
                mismatches.append(f"{group_path}.parameter_names: base group must exclude 'head.query_token'")
            if "head.gate_logit" not in parameter_names:
                mismatches.append(f"{group_path}.parameter_names: base group must include 'head.gate_logit'")
        elif list(parameter_names) != ["head.query_token"]:
            mismatches.append(f"{group_path}.parameter_names: query group must contain only 'head.query_token'")
    return mismatches


def _validate_last_tuned_resolved_protocol(
    experiment: Any,
    optimizer_config: Mapping[str, Any],
) -> None:
    """Fail closed unless the resolved tuning run matches its fixed contract."""

    if not isinstance(optimizer_config, Mapping):
        raise ProtocolMismatchError("last_tuned optimizer_config must be a mapping")

    experiment_checks = {
        "trainer_config.n_epochs": PROTOCOL_CONTRACT["max_epochs"],
        "trainer_config.monitor": PROTOCOL_CONTRACT["monitor"],
        "trainer_config.mode": PROTOCOL_CONTRACT["mode"],
        "trainer_config.patience": PROTOCOL_CONTRACT["patience"],
        "trainer_config.gradient_clip_val": PROTOCOL_CONTRACT["gradient_clip_val"],
        "trainer_config.precision": PROTOCOL_CONTRACT["precision"],
        "loss.name": PROTOCOL_CONTRACT["loss"],
        "data.batch_size": PROTOCOL_CONTRACT["batch_size"],
        "data.num_workers": PROTOCOL_CONTRACT["batch_num_workers"],
        "data.persistent_workers": PROTOCOL_CONTRACT["persistent_workers"],
        "data.seed": 33,
        "data.duration": PROTOCOL_CONTRACT["window_duration_s"],
        "data.stride": PROTOCOL_CONTRACT["window_stride_s"],
        "data.neuro.frequency": PROTOCOL_CONTRACT["frequency"],
        "data.neuro.filter": PROTOCOL_CONTRACT["filter"],
        "data.neuro.notch_filter": PROTOCOL_CONTRACT["notch_filter"],
        "data.neuro.scaler": PROTOCOL_CONTRACT["scaler"],
        "data.neuro.clamp": PROTOCOL_CONTRACT["clamp"],
        "target_scaler": PROTOCOL_CONTRACT["target_scaler"],
        "checkpoint_selection.monitor": PROTOCOL_CONTRACT["monitor"],
        "checkpoint_selection.mode": PROTOCOL_CONTRACT["mode"],
        "checkpoint_selection.test_pearsonr_role": "diagnostic_only",
    }
    optimizer_checks = {
        "optimizer.name": PROTOCOL_CONTRACT["optimizer"],
        "optimizer.lr": LAST_TUNED_BASE_LR,
        "optimizer.kwargs.weight_decay": LAST_TUNED_WEIGHT_DECAY,
        "scheduler.name": PROTOCOL_CONTRACT["scheduler"],
        "scheduler.kwargs.max_lr": list(LAST_TUNED_SCHEDULER_MAX_LR),
        "scheduler.kwargs.pct_start": LAST_TUNED_SCHEDULER_PCT_START,
        "scheduler.kwargs.anneal_strategy": PROTOCOL_CONTRACT[
            "scheduler_anneal_strategy"
        ],
        "scheduler.kwargs.div_factor": LAST_TUNED_SCHEDULER_DIV_FACTOR,
        "scheduler.kwargs.final_div_factor": LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
        "scheduler.interval": "step",
        "scheduler.frequency": 1,
    }
    mismatches = _path_mismatches(experiment, experiment_checks)
    mismatches.extend(_path_mismatches(optimizer_config, optimizer_checks, prefix="optimizer_config."))
    mismatches.extend(_last_tuned_group_mismatches(_get_path(optimizer_config, "param_groups")))

    if mismatches:
        raise ProtocolMismatchError(
            "resolved last_tuned protocol does not match the fixed tuning contract:\n"
            + "\n".join(f"- {item}" for item in mismatches)
        )


def validate_official_protocol(
    experiment: Any,
    *,
    loaders: Mapping[str, Any] | None = None,
    n_total_params: int | None = None,
    n_trainable_params: int | None = None,
    allow_target_scaler: bool = False,
) -> None:
    """Fail before fit if the resolved NeuralBench protocol is not canonical."""

    checks = {
        "trainer_config.n_epochs": PROTOCOL_CONTRACT["max_epochs"],
        "trainer_config.monitor": PROTOCOL_CONTRACT["monitor"],
        "trainer_config.mode": PROTOCOL_CONTRACT["mode"],
        "trainer_config.patience": PROTOCOL_CONTRACT["patience"],
        "trainer_config.gradient_clip_val": PROTOCOL_CONTRACT["gradient_clip_val"],
        "trainer_config.precision": PROTOCOL_CONTRACT["precision"],
        "lightning_optimizer_config.optimizer.name": PROTOCOL_CONTRACT["optimizer"],
        "lightning_optimizer_config.optimizer.lr": PROTOCOL_CONTRACT["learning_rate"],
        "lightning_optimizer_config.optimizer.kwargs.weight_decay": PROTOCOL_CONTRACT[
            "weight_decay"
        ],
        "lightning_optimizer_config.scheduler.name": PROTOCOL_CONTRACT["scheduler"],
        "lightning_optimizer_config.scheduler.kwargs.max_lr": PROTOCOL_CONTRACT[
            "scheduler_max_lr"
        ],
        "lightning_optimizer_config.scheduler.kwargs.pct_start": PROTOCOL_CONTRACT[
            "scheduler_pct_start"
        ],
        "lightning_optimizer_config.scheduler.kwargs.anneal_strategy": PROTOCOL_CONTRACT[
            "scheduler_anneal_strategy"
        ],
        "loss.name": PROTOCOL_CONTRACT["loss"],
        "data.batch_size": PROTOCOL_CONTRACT["batch_size"],
        "data.num_workers": PROTOCOL_CONTRACT["batch_num_workers"],
        "data.persistent_workers": PROTOCOL_CONTRACT["persistent_workers"],
        "data.seed": 33,
        "data.duration": PROTOCOL_CONTRACT["window_duration_s"],
        "data.stride": PROTOCOL_CONTRACT["window_stride_s"],
        "data.neuro.frequency": PROTOCOL_CONTRACT["frequency"],
        "data.neuro.filter": PROTOCOL_CONTRACT["filter"],
        "data.neuro.notch_filter": PROTOCOL_CONTRACT["notch_filter"],
        "data.neuro.scaler": PROTOCOL_CONTRACT["scaler"],
        "data.neuro.clamp": PROTOCOL_CONTRACT["clamp"],
        "target_scaler": PROTOCOL_CONTRACT["target_scaler"],
    }
    mismatches: list[str] = []
    for path, expected in checks.items():
        actual = _get_path(experiment, path)
        if path == "target_scaler" and allow_target_scaler:
            if actual is _MISSING or actual is None:
                mismatches.append("target_scaler: expected fitted training-only StandardScaler")
            continue
        if actual is _MISSING or not _same(actual, expected):
            mismatches.append(f"{path}: expected {expected!r}, got {actual!r}")

    if loaders is not None:
        for split, loader in loaders.items():
            if getattr(loader, "batch_size", _MISSING) != PROTOCOL_CONTRACT["batch_size"]:
                mismatches.append(f"{split}.loader.batch_size is not 64")
            if getattr(loader, "num_workers", _MISSING) != PROTOCOL_CONTRACT["batch_num_workers"]:
                mismatches.append(f"{split}.loader.num_workers is not 2")
            if getattr(loader, "persistent_workers", _MISSING) != PROTOCOL_CONTRACT[
                "persistent_workers"
            ]:
                mismatches.append(f"{split}.loader.persistent_workers is not True")

    wrapper = _get_path(experiment, "downstream_model_wrapper")
    if wrapper is not _MISSING and wrapper is not None:
        for field in (
            "on_the_fly_preprocessor",
            "channel_adapter_config",
            "model_output_key",
        ):
            if _get_path(wrapper, field) not in (None, _MISSING):
                mismatches.append(f"downstream_model_wrapper.{field} must be None")
        for field in ("layers_to_freeze", "layers_to_unfreeze"):
            if _get_path(wrapper, field) not in (None, _MISSING):
                mismatches.append(f"downstream_model_wrapper.{field} must be None for full fine-tuning")

    if n_total_params is not None and n_trainable_params != n_total_params:
        mismatches.append(f"backbone trainability: expected all parameters trainable, got {n_trainable_params}/{n_total_params}")

    if mismatches:
        raise ProtocolMismatchError(
            "resolved NeuralBench protocol does not match the canonical contract:\n"
            + "\n".join(f"- {item}" for item in mismatches)
        )


def verify_upstream_source_hashes(source_root: Path) -> dict[str, str]:
    """Verify a local checkout matches the pinned upstream source snapshot."""

    observed: dict[str, str] = {}
    for relative, expected in UPSTREAM_REVE_FILE_HASHES.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned upstream source file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[relative] = digest
        if digest != expected:
            raise ValueError(f"upstream source hash mismatch for {relative}: expected {expected}, got {digest}")
    return observed


def source_lock_metadata() -> dict[str, Any]:
    """Return immutable source-lock fields for experiment artifacts."""

    return {
        "repository": UPSTREAM_REVE_REPOSITORY,
        "commit": UPSTREAM_REVE_COMMIT,
        "file_sha256": dict(UPSTREAM_REVE_FILE_HASHES),
    }


def runtime_metadata() -> dict[str, Any]:
    """Collect reproducibility metadata without importing heavy modules."""

    package_versions: dict[str, str | None] = {}
    for package in (
        "neuralbench",
        "neuraltrain",
        "braindecode",
        "lightning",
        "torch",
        "neuralfetch",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    source_locations: dict[str, str | None] = {}
    for package in ("neuralbench", "neuraltrain"):
        spec = importlib.util.find_spec(package)
        source_locations[package] = None if spec is None else spec.origin

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "source_locations": source_locations,
    }
