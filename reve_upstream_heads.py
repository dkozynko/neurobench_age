"""Exact upstream REVE downstream heads and the NeuralBench adapter.

The pure head implementation in this module follows the pinned upstream REVE
classifier.  The optional NeuralBench adapter is created lazily so the head
math can be tested without importing the full NeuralBench/NeuralTrain stack.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import math
import platform
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
from torch import nn


UPSTREAM_REVE_REPOSITORY = "https://github.com/elouayas/reve_eeg"
UPSTREAM_REVE_COMMIT = "06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7"
UPSTREAM_REVE_FILE_HASHES = {
    "src/models/classifier.py": "cf71d0f455df0e9a263363213d75fbf493ffc5d587cd3c1f80cdc90ab961f90f",
    "src/models/encoder.py": "933684a3306e16c16926cead0fc8c8c0a9ae99ad9dfaf54dee860c5b335d1b6c",
    "src/models/backbone.py": "632ecd10ddb5efcc6f50a7207b0aa3934e7304438eb3f1db1edfefc3fab3371f",
}

OFFICIAL_HEAD_VARIANTS = ("mean_linear", "last_avg", "last", "all")
LAST_TUNED_PROTOCOL_VARIANTS = ("last_tuned",)
HEAD_VARIANTS = OFFICIAL_HEAD_VARIANTS + LAST_TUNED_PROTOCOL_VARIANTS
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
        raise ValueError(
            f"unsupported REVE head variant {variant!r}; "
            f"expected one of {HEAD_VARIANTS}"
        )
    return variant


def validate_official_head_variant(variant: str) -> str:
    """Validate a head variant admitted by the official protocol."""

    if variant not in OFFICIAL_HEAD_VARIANTS:
        raise ValueError(
            f"{variant!r} is not an official REVE head variant; "
            f"expected one of {OFFICIAL_HEAD_VARIANTS}"
        )
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
        raise ValueError(
            f"{variant!r} is not a last_tuned protocol variant; "
            f"expected one of {LAST_TUNED_PROTOCOL_VARIANTS}"
        )
    if experiment is None and optimizer_config is None:
        return variant
    if experiment is None or optimizer_config is None:
        raise ProtocolMismatchError(
            "last_tuned resolved validation requires both experiment and "
            "optimizer_config"
        )
    _validate_last_tuned_resolved_protocol(experiment, optimizer_config)
    return variant


def validate_upstream_head_variant(variant: str) -> str:
    """Validate and return a non-baseline upstream head variant."""

    validate_official_head_variant(variant)
    if variant not in UPSTREAM_HEAD_VARIANTS:
        raise ValueError(
            f"{variant!r} is the official baseline, not an upstream head variant"
        )
    return variant


class RMSNorm(nn.Module):
    """The RMSNorm used by the pinned upstream REVE classifier."""

    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def _reject_mask_kwargs(kwargs: Mapping[str, Any]) -> None:
    mask_keys = {
        "attention_mask",
        "key_padding_mask",
        "mask",
        "padding_mask",
    }
    supplied = sorted(mask_keys.intersection(kwargs))
    if supplied:
        raise AdapterContractError(
            "the fixed-window REVE adapter does not accept a required mask; "
            f"received {supplied}"
        )


def _validate_final_tokens(tokens: Any, *, embed_dim: int) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise AdapterContractError(
            "last/last_avg require a final token tensor with shape [batch, tokens, dim]"
        )
    if tokens.shape[-1] != embed_dim:
        raise AdapterContractError(
            f"final token tensor has embedding dimension {tokens.shape[-1]}, "
            f"expected {embed_dim}"
        )
    if tokens.shape[1] <= 0:
        raise AdapterContractError("final token tensor must contain at least one token")
    return tokens


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Return a device-independent SHA-256 digest for a tensor's raw values."""

    return hashlib.sha256(
        tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _encoder_primary_input_key(encoder: nn.Module) -> str:
    """Return the concrete first input name used by NeuralBench model_factory."""

    for parameter in inspect.signature(encoder.forward).parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return parameter.name
    raise AdapterContractError("could not derive the encoder primary input key")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise AdapterContractError(f"{name} must be a scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterContractError(f"{name} must be a positive integer")
    return value


def _expected_reve_token_count(encoder: nn.Module, eeg: torch.Tensor) -> int:
    """Derive REVE's dense patch-token count from its patch geometry."""

    if eeg.ndim != 3 or eeg.shape[0] != 1:
        raise AdapterContractError("last_tuned dummy EEG must have shape [1, C, T]")
    underlying = _unwrap_reve_module(encoder)
    patch_size = _positive_int(getattr(underlying, "patch_size", None), name="patch_size")
    patch_overlap = getattr(underlying, "patch_overlap", None)
    if isinstance(patch_overlap, torch.Tensor):
        if patch_overlap.numel() != 1:
            raise AdapterContractError("patch_overlap must be a scalar")
        patch_overlap = patch_overlap.item()
    if isinstance(patch_overlap, bool) or not isinstance(patch_overlap, int):
        raise AdapterContractError("patch_overlap must be an integer")
    stride = patch_size - patch_overlap
    if stride <= 0:
        raise AdapterContractError("patch_overlap must be smaller than patch_size")

    samples = int(eeg.shape[2])
    if samples < patch_size:
        raise AdapterContractError("input is shorter than one REVE patch")
    channel_indices = getattr(encoder, "channel_indices", None)
    if channel_indices is None:
        channel_count = int(eeg.shape[1])
    elif isinstance(channel_indices, torch.Tensor):
        channel_count = int(channel_indices.numel())
    elif isinstance(channel_indices, (list, tuple)):
        channel_count = len(channel_indices)
    else:
        raise AdapterContractError("REVE channel_indices must be a tensor or sequence")
    if channel_count <= 0:
        raise AdapterContractError("REVE must retain at least one input channel")

    n_patches = 1 + (samples - patch_size) // stride
    return channel_count * n_patches


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def initialize_last_tuned_query(
    encoder: nn.Module,
    dummy_batch: Mapping[str, Any],
    *,
    provenance: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Initialize ``last_tuned`` from NeuralBench's train-dummy batch only."""

    if provenance is not _NEURALBENCH_TRAIN_DUMMY_CONTEXT:
        raise AdapterContractError(
            "last_tuned initialization requires the internal train-dummy provenance context"
        )
    if not isinstance(dummy_batch, Mapping):
        raise AdapterContractError("last_tuned dummy batch must be a mapping")

    input_key = _encoder_primary_input_key(encoder)
    allowed_keys = {input_key, "channel_positions", "subject_ids"}
    keys = set(dummy_batch)
    if input_key not in keys or not keys <= allowed_keys:
        raise AdapterContractError(
            "last_tuned dummy batch must contain the encoder primary input and optional "
            "channel_positions or subject_ids"
        )

    eeg = dummy_batch[input_key]
    if not isinstance(eeg, torch.Tensor):
        raise AdapterContractError("last_tuned dummy batch contains no EEG tensor")
    expected_token_count = _expected_reve_token_count(encoder, eeg)

    channel_positions = dummy_batch.get("channel_positions")
    if channel_positions is not None and not isinstance(channel_positions, torch.Tensor):
        raise AdapterContractError("last_tuned channel positions must be a tensor or None")

    modules = tuple(encoder.modules())
    training_flags = tuple(module.training for module in modules)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        encoder.eval()
        with torch.inference_mode():
            with torch.autocast(device_type=eeg.device.type, enabled=False):
                output = (
                    encoder(eeg)
                    if channel_positions is None
                    else encoder(eeg, pos=channel_positions)
                )
                if not isinstance(output, torch.Tensor):
                    raise AdapterContractError(
                        "last_tuned encoder must return a final-token tensor"
                    )
                if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] <= 0:
                    raise AdapterContractError(
                        "last_tuned encoder output must have shape [1, T, D] with T > 0"
                    )
                if output.shape[1] != expected_token_count:
                    raise AdapterContractError(
                        "last_tuned encoder token count does not match REVE patch geometry"
                    )
                if output.shape[2] <= 0:
                    raise AdapterContractError("last_tuned encoder output embed dim must be positive")
                if not torch.isfinite(output).all():
                    raise AdapterContractError(
                        "last_tuned final token tensor must contain only finite values"
                    )
                mean_query = output.mean(dim=1, keepdim=True)
                if not torch.isfinite(mean_query).all():
                    raise AdapterContractError(
                        "last_tuned train-dummy mean query must contain only finite values"
                    )
        query = mean_query.detach().clone()
    finally:
        for module, was_training in zip(modules, training_flags):
            module.training = was_training
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    metadata = {
        "query_initialization": "train_dummy_final_token_mean",
        "query_initialization_batch_element": 0,
        "query_initialization_input_shape": list(eeg.shape),
        "query_initialization_input_dtype": str(eeg.dtype),
        "query_initialization_input_device": str(eeg.device),
        "query_initialization_input_sha256": _tensor_sha256(eeg),
        "query_initialization_query_sha256": _tensor_sha256(query),
    }
    if "subject_ids" in dummy_batch:
        metadata["query_initialization_subject_ids"] = _json_safe_value(
            dummy_batch["subject_ids"]
        )
    return query, metadata


def _validate_query_token(query_token: Any, *, embed_dim: int) -> torch.Tensor:
    if not isinstance(query_token, torch.Tensor) or tuple(query_token.shape) != (
        1,
        1,
        embed_dim,
    ):
        shape = getattr(query_token, "shape", None)
        raise AdapterContractError(
            f"cls_query_token must have shape [1, 1, {embed_dim}], got {shape}"
        )
    if not query_token.is_floating_point():
        raise AdapterContractError("cls_query_token must have a floating-point dtype")
    if not torch.isfinite(query_token).all():
        raise AdapterContractError("cls_query_token must contain only finite values")
    return query_token


def _last_tuned_total_steps(trainer: Any) -> int:
    """Return Lightning's resolved optimizer-step count, failing closed."""

    if trainer is None:
        raise ProtocolMismatchError("last_tuned optimizer requires a Lightning trainer")
    total_steps = getattr(trainer, "estimated_stepping_batches", None)
    if (
        isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps <= 0
    ):
        raise ProtocolMismatchError(
            "trainer.estimated_stepping_batches must be a positive integer"
        )
    return total_steps


def _resolve_last_tuned_model(model: nn.Module) -> nn.Module:
    """Resolve the REVE model that owns ``head.query_token``.

    NeuralBench wraps a downstream model in ``DownstreamWrapperModel`` after
    the adapter is built.  That wrapper keeps the actual model under
    ``wrapped_model`` and adds only identity aggregation/probe modules for
    this protocol.  Pure optimizer tests pass the underlying model directly,
    so accept both shapes while keeping parameter names normalized to the
    underlying REVE model.
    """

    if not isinstance(model, nn.Module):
        raise ProtocolMismatchError("last_tuned optimizer model must be an nn.Module")
    current = model
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if getattr(current, "head", None) is not None:
            return current
        inner = getattr(current, "wrapped_model", None)
        if isinstance(inner, nn.Module):
            current = inner
            continue
        break
    raise ProtocolMismatchError(
        "last_tuned optimizer requires a model with a REVE head under "
        "head or wrapped_model.head"
    )


def _last_tuned_trainable_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], nn.Parameter]:
    """Resolve exact base/query membership from deterministic parameter names."""

    model = _resolve_last_tuned_model(model)
    head = getattr(model, "head", None)
    query = getattr(head, "query_token", None)
    if not isinstance(query, nn.Parameter):
        raise ProtocolMismatchError(
            "last_tuned optimizer requires model.head.query_token"
        )
    if not query.requires_grad:
        raise ProtocolMismatchError("model.head.query_token must be trainable")

    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    ]
    if not named_trainable:
        raise ProtocolMismatchError("last_tuned model has no trainable parameters")

    names_by_identity: dict[int, str] = {}
    for name, parameter in named_trainable:
        identity = id(parameter)
        previous_name = names_by_identity.get(identity)
        if previous_name is not None:
            raise ProtocolMismatchError(
                "duplicate trainable parameter identity registered as "
                f"{previous_name!r} and {name!r}"
            )
        names_by_identity[identity] = name

    query_entries = [
        (name, parameter)
        for name, parameter in named_trainable
        if parameter is query
    ]
    if len(query_entries) != 1 or query_entries[0][0] != "head.query_token":
        observed = [name for name, _parameter in query_entries]
        raise ProtocolMismatchError(
            "last_tuned query must be registered exactly once as "
            f"'head.query_token'; got {observed!r}"
        )

    base_parameters = [
        parameter for _name, parameter in named_trainable if parameter is not query
    ]
    if not base_parameters:
        raise ProtocolMismatchError(
            "last_tuned base optimizer group must contain trainable parameters"
        )

    resolved_parameters = base_parameters + [query]
    resolved_identities = [id(parameter) for parameter in resolved_parameters]
    model_trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    model_identities = [id(parameter) for parameter in model_trainable]
    if len(model_identities) != len(set(model_identities)):
        raise ProtocolMismatchError("model.parameters() returned duplicate identities")
    if set(resolved_identities) != set(model_identities):
        missing = set(model_identities) - set(resolved_identities)
        unexpected = set(resolved_identities) - set(model_identities)
        raise ProtocolMismatchError(
            "last_tuned optimizer parameter resolution mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    return base_parameters, query


def _validate_last_tuned_optimizer_config(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    base_parameters: Sequence[nn.Parameter],
    query: nn.Parameter,
    total_steps: int,
) -> None:
    """Validate the complete differential optimizer contract before fit."""

    model = _resolve_last_tuned_model(model)

    if type(optimizer) is not torch.optim.AdamW:
        raise ProtocolMismatchError("last_tuned optimizer must be exactly AdamW")
    if type(scheduler) is not torch.optim.lr_scheduler.OneCycleLR:
        raise ProtocolMismatchError("last_tuned scheduler must be exactly OneCycleLR")
    if len(optimizer.param_groups) != 2:
        raise ProtocolMismatchError("last_tuned optimizer must have exactly two groups")
    if scheduler.optimizer is not optimizer or scheduler.total_steps != total_steps:
        raise ProtocolMismatchError("last_tuned OneCycleLR total-step binding is invalid")
    if scheduler._anneal_func_type != "cos":
        raise ProtocolMismatchError("last_tuned OneCycleLR must use cosine annealing")

    expected_parameters = (list(base_parameters), [query])
    expected_max_lrs = LAST_TUNED_SCHEDULER_MAX_LR
    expected_initial_lrs = tuple(
        max_lr / LAST_TUNED_SCHEDULER_DIV_FACTOR
        for max_lr in expected_max_lrs
    )
    expected_min_lrs = tuple(
        initial_lr / LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR
        for initial_lr in expected_initial_lrs
    )
    observed_identities: list[int] = []
    for index, (
        group,
        parameters,
        max_lr,
        initial_lr,
        min_lr,
    ) in enumerate(
        zip(
            optimizer.param_groups,
            expected_parameters,
            expected_max_lrs,
            expected_initial_lrs,
            expected_min_lrs,
        )
    ):
        observed_parameters = list(group["params"])
        if [id(parameter) for parameter in observed_parameters] != [
            id(parameter) for parameter in parameters
        ]:
            raise ProtocolMismatchError(
                f"last_tuned optimizer group {index} parameter membership changed"
            )
        observed_identities.extend(id(parameter) for parameter in observed_parameters)
        expected_values = {
            "weight_decay": LAST_TUNED_WEIGHT_DECAY,
            "initial_lr": initial_lr,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "lr": initial_lr,
        }
        for field, expected in expected_values.items():
            actual = group.get(field)
            if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ProtocolMismatchError(
                    f"last_tuned optimizer group {index} {field} mismatch: "
                    f"expected {expected!r}, got {actual!r}"
                )

    trainable_identities = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if len(observed_identities) != len(set(observed_identities)):
        raise ProtocolMismatchError("last_tuned optimizer contains duplicate parameters")
    if set(observed_identities) != trainable_identities:
        missing = trainable_identities - set(observed_identities)
        unexpected = set(observed_identities) - trainable_identities
        raise ProtocolMismatchError(
            "last_tuned optimizer trainable-parameter mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    expected_first_phase_end = total_steps * LAST_TUNED_SCHEDULER_PCT_START - 1
    phases = getattr(scheduler, "_schedule_phases", ())
    if not phases or not math.isclose(
        float(phases[0].get("end_step", float("nan"))),
        expected_first_phase_end,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ProtocolMismatchError("last_tuned OneCycleLR pct_start is invalid")


def build_last_tuned_optimizer_config(
    model: nn.Module,
    trainer: Any,
) -> dict[str, Any]:
    """Build the exact two-group AdamW/OneCycleLR tuning configuration."""

    model = _resolve_last_tuned_model(model)
    total_steps = _last_tuned_total_steps(trainer)
    base_parameters, query = _last_tuned_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        (
            {
                "params": base_parameters,
                "lr": LAST_TUNED_BASE_LR,
                "weight_decay": LAST_TUNED_WEIGHT_DECAY,
            },
            {
                "params": [query],
                "lr": LAST_TUNED_QUERY_LR,
                "weight_decay": LAST_TUNED_WEIGHT_DECAY,
            },
        ),
        lr=LAST_TUNED_BASE_LR,
        weight_decay=LAST_TUNED_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=list(LAST_TUNED_SCHEDULER_MAX_LR),
        total_steps=total_steps,
        pct_start=LAST_TUNED_SCHEDULER_PCT_START,
        anneal_strategy="cos",
        div_factor=LAST_TUNED_SCHEDULER_DIV_FACTOR,
        final_div_factor=LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
    )
    _validate_last_tuned_optimizer_config(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        base_parameters=base_parameters,
        query=query,
        total_steps=total_steps,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }


def last_tuned_optimizer_metadata(model: nn.Module) -> dict[str, Any]:
    """Describe the fixed tuning optimizer without constructing it.

    This shares the production parameter-resolution seam with
    :func:`build_last_tuned_optimizer_config`, so the metadata used for the
    protocol gate and report cannot conceal a missing, duplicated, or
    mis-grouped trainable parameter.
    """

    model = _resolve_last_tuned_model(model)
    base_parameters, query = _last_tuned_trainable_parameters(model)
    names_by_identity = {
        id(parameter): name
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    base_names = [names_by_identity.get(id(parameter)) for parameter in base_parameters]
    query_name = names_by_identity.get(id(query))
    if any(name is None for name in base_names) or query_name != "head.query_token":
        raise ProtocolMismatchError(
            "last_tuned optimizer metadata could not resolve deterministic parameter names"
        )

    return {
        "optimizer": {
            "name": "AdamW",
            "lr": LAST_TUNED_BASE_LR,
            "kwargs": {"weight_decay": LAST_TUNED_WEIGHT_DECAY},
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {
                "max_lr": list(LAST_TUNED_SCHEDULER_MAX_LR),
                "pct_start": LAST_TUNED_SCHEDULER_PCT_START,
                "anneal_strategy": "cos",
                "div_factor": LAST_TUNED_SCHEDULER_DIV_FACTOR,
                "final_div_factor": LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
            },
            "interval": "step",
            "frequency": 1,
        },
        "param_groups": [
            {
                "name": "base",
                "parameter_names": base_names,
                "learning_rate": LAST_TUNED_BASE_LR,
                "weight_decay": LAST_TUNED_WEIGHT_DECAY,
            },
            {
                "name": "query",
                "parameter_names": [query_name],
                "learning_rate": LAST_TUNED_QUERY_LR,
                "weight_decay": LAST_TUNED_WEIGHT_DECAY,
            },
        ],
    }


def concatenate_all_layers(
    layers: Sequence[torch.Tensor], *, embed_dim: int | None = None
) -> torch.Tensor:
    """Concatenate REVE's positional input and transformer outputs by tokens."""

    if isinstance(layers, torch.Tensor) or not isinstance(layers, (list, tuple)):
        raise AdapterContractError(
            "all requires an ordered sequence of [batch, tokens, dim] tensors"
        )
    if not layers:
        raise AdapterContractError("all requires at least one encoder output sequence")

    first = layers[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 3:
        raise AdapterContractError(
            "all requires every encoder output to have shape [batch, tokens, dim]"
        )
    expected_shape = (first.shape[0], first.shape[-1])
    if embed_dim is not None and first.shape[-1] != embed_dim:
        raise AdapterContractError(
            f"all output has embedding dimension {first.shape[-1]}, expected {embed_dim}"
        )
    for index, layer in enumerate(layers):
        if not isinstance(layer, torch.Tensor) or layer.ndim != 3:
            raise AdapterContractError(
                "all requires every encoder output to have shape "
                f"[batch, tokens, dim]; item {index} is invalid"
            )
        if (layer.shape[0], layer.shape[-1]) != expected_shape:
            raise AdapterContractError(
                "all encoder outputs must share batch and embedding dimensions; "
                f"item {index} has shape {tuple(layer.shape)}"
            )
        if layer.shape[1] <= 0:
            raise AdapterContractError(f"all encoder output {index} has no tokens")
    return torch.cat(tuple(layers), dim=1)


class UpstreamReveHead(nn.Module):
    """Upstream REVE heads plus the isolated pure ``last_tuned`` head."""

    def __init__(
        self,
        *,
        variant: str,
        embed_dim: int,
        n_outputs: int,
        dropout: float = DEFAULT_UPSTREAM_DROPOUT,
        query_token: torch.Tensor | None = None,
        query_initialization_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if variant == "last_tuned":
            validate_last_tuned_protocol(variant)
        else:
            validate_upstream_head_variant(variant)
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if variant == "last_tuned":
            if dropout != DEFAULT_UPSTREAM_DROPOUT:
                raise ValueError("last_tuned fixes dropout=0.0")
            if query_token is None:
                raise AdapterContractError("last_tuned requires an explicit cls_query_token")
            query_token = _validate_query_token(query_token, embed_dim=embed_dim)
        elif query_token is not None and tuple(query_token.shape) != (1, 1, embed_dim):
            raise AdapterContractError(
                f"cls_query_token must have shape [1, 1, {embed_dim}], "
                f"got {tuple(query_token.shape)}"
            )

        self.variant = variant
        self.embed_dim = embed_dim
        self.n_outputs = n_outputs
        # The pinned upstream ReveClassifier creates this downstream query
        # token with torch.randn for every pooling mode; ``last_avg`` simply
        # does not use it in forward. It is not part of the pretrained encoder
        # checkpoint. NeuralBench seeds the model build before this wrapper is
        # constructed, so this remains reproducible per experiment seed.
        if query_token is None:
            query_token = torch.randn(1, 1, embed_dim)
            self.query_initialization = (
                "upstream_random_unused" if variant == "last_avg" else "upstream_random"
            )
        else:
            self.query_initialization = "provided"
        self.query_token = nn.Parameter(query_token.detach().clone())
        if variant == "last_tuned":
            self.gate_logit = nn.Parameter(
                torch.tensor(
                    math.log(LAST_TUNED_INITIAL_ALPHA / (1.0 - LAST_TUNED_INITIAL_ALPHA))
                )
            )
            tuning_metadata: dict[str, Any] = {
                "head_variant": variant,
                "head_source": LAST_TUNED_HEAD_SOURCE,
                "head_architecture": LAST_TUNED_HEAD_ARCHITECTURE,
                "protocol_class": LAST_TUNED_PROTOCOL_CLASS,
                "residual_initial_alpha": LAST_TUNED_INITIAL_ALPHA,
                "query_initialization": self.query_initialization,
            }
            if query_initialization_metadata is not None:
                tuning_metadata.update(copy.deepcopy(dict(query_initialization_metadata)))
                query_initialization = tuning_metadata.get("query_initialization")
                if not isinstance(query_initialization, str):
                    raise AdapterContractError(
                        "last_tuned query initialization metadata must name its source"
                    )
                self.query_initialization = query_initialization
            tuning_metadata["query_initialization"] = self.query_initialization
            self._tuning_metadata = MappingProxyType(copy.deepcopy(tuning_metadata))
        self.dropout = nn.Dropout(float(dropout))
        self.norm = RMSNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, n_outputs)
        # ``config/init/cls_wrapper.yaml`` makes the upstream classifier's
        # final linear head use truncated normal(std=hidden_size**-0.5) and a
        # zero bias. Keep the default Linear construction before overriding it
        # to preserve the upstream RNG/parameter construction order.
        nn.init.trunc_normal_(
            self.linear.weight,
            mean=0.0,
            std=UPSTREAM_HEAD_INIT_STD,
            a=-UPSTREAM_HEAD_INIT_CUTOFF * UPSTREAM_HEAD_INIT_STD,
            b=UPSTREAM_HEAD_INIT_CUTOFF * UPSTREAM_HEAD_INIT_STD,
        )
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def _validate_last_tuned_state(self) -> None:
        if not torch.isfinite(self.query_token).all():
            raise AdapterContractError("cls_query_token must contain only finite values")
        if not torch.isfinite(self.gate_logit).all():
            raise AdapterContractError("last_tuned gate_logit must be finite")

    @property
    def residual_alpha(self) -> float:
        """The learned last_tuned interpolation weight as a detached scalar."""

        if self.variant != "last_tuned":
            raise AttributeError("residual_alpha is only defined for last_tuned")
        self._validate_last_tuned_state()
        return float(torch.sigmoid(self.gate_logit).detach())

    @property
    def residual_initial_alpha(self) -> float:
        """The immutable initial interpolation weight for last_tuned."""

        if self.variant != "last_tuned":
            raise AttributeError("residual_initial_alpha is only defined for last_tuned")
        return LAST_TUNED_INITIAL_ALPHA

    @property
    def protocol_class(self) -> str:
        """The immutable protocol classification for last_tuned."""

        if self.variant != "last_tuned":
            raise AttributeError("protocol_class is only defined for last_tuned")
        return LAST_TUNED_PROTOCOL_CLASS

    @property
    def head_source(self) -> str:
        """The immutable source label for last_tuned."""

        if self.variant != "last_tuned":
            raise AttributeError("head_source is only defined for last_tuned")
        return LAST_TUNED_HEAD_SOURCE

    @property
    def head_architecture(self) -> str:
        """The immutable architecture label for last_tuned."""

        if self.variant != "last_tuned":
            raise AttributeError("head_architecture is only defined for last_tuned")
        return LAST_TUNED_HEAD_ARCHITECTURE

    @property
    def tuning_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the tuned head metadata."""

        if self.variant != "last_tuned":
            raise AttributeError("tuning_metadata is only defined for last_tuned")
        return copy.deepcopy(dict(self._tuning_metadata))

    @staticmethod
    def _test_alpha_override(
        _test_alpha: float | torch.Tensor, *, reference: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(_test_alpha, torch.Tensor):
            if _test_alpha.numel() != 1:
                raise AdapterContractError("last_tuned alpha override must be scalar")
            _test_alpha = _test_alpha.to(dtype=reference.dtype, device=reference.device)
        else:
            try:
                _test_alpha = reference.new_tensor(float(_test_alpha))
            except (TypeError, ValueError) as error:
                raise AdapterContractError(
                    "last_tuned _test_alpha override must be a finite scalar in [0, 1]"
                ) from error
        if not torch.isfinite(_test_alpha).all():
            raise AdapterContractError("last_tuned _test_alpha override must be finite")
        if not bool(((_test_alpha >= 0) & (_test_alpha <= 1)).all()):
            raise AdapterContractError("last_tuned _test_alpha override must be in [0, 1]")
        return _test_alpha

    def pool_tokens(
        self,
        tokens: Any,
        *,
        _test_alpha: float | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the representation immediately before RMSNorm/dropout/linear."""

        _reject_mask_kwargs(kwargs)
        if self.variant == "all":
            tokens = concatenate_all_layers(tokens, embed_dim=self.embed_dim)
        else:
            tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)

        if self.variant == "last_avg":
            return tokens.mean(dim=1)

        assert self.query_token is not None
        if self.variant == "last_tuned":
            if not torch.isfinite(tokens).all():
                raise AdapterContractError(
                    "last_tuned final token tensor must contain only finite values"
                )
            self._validate_last_tuned_state()
            query = self.query_token.expand(tokens.shape[0], -1, -1)
            scores = torch.einsum("bqd,btd->bqt", query, tokens) / math.sqrt(
                self.embed_dim
            )
            if not torch.isfinite(scores).all():
                raise AdapterContractError("last_tuned attention scores must be finite")
            weights = torch.softmax(scores, dim=-1)
            if not torch.isfinite(weights).all():
                raise AdapterContractError("last_tuned attention weights must be finite")
            attention = torch.einsum("bqt,btd->bqd", weights, tokens).squeeze(1)
            if not torch.isfinite(attention).all():
                raise AdapterContractError("last_tuned attention context must be finite")
            mean = tokens.mean(dim=1)
            if not torch.isfinite(mean).all():
                raise AdapterContractError("last_tuned token mean must be finite")
            residual_alpha = (
                torch.sigmoid(self.gate_logit)
                if _test_alpha is None
                else self._test_alpha_override(_test_alpha, reference=mean)
            )
            mixed = mean + residual_alpha * (attention - mean)
            if not torch.isfinite(mixed).all():
                raise AdapterContractError("last_tuned mixed residual must be finite")
            return mixed

        query = self.query_token.expand(tokens.shape[0], -1, -1)
        scores = torch.matmul(query, tokens.transpose(-1, -2)) / (self.embed_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, tokens).squeeze(1)

    def forward(self, tokens: Any, **kwargs: Any) -> torch.Tensor:
        representation = self.pool_tokens(tokens, **kwargs)
        return self.linear(self.dropout(self.norm(representation)))


def _unwrap_reve_module(module: nn.Module) -> nn.Module:
    """Find the underlying braindecode REVE module inside `_ReveWrapper`."""

    current = module
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        inner = getattr(current, "model", None)
        if not isinstance(inner, nn.Module):
            return current
        current = inner
    raise AdapterContractError("cyclic REVE wrapper structure")


class AllLayerReveEncoder:
    """Expose the exact all-layer output of NeuralTrain's `_ReveWrapper`."""

    def __init__(self, wrapped_encoder: nn.Module):
        self.wrapped_encoder = wrapped_encoder

    def __call__(
        self,
        eeg: torch.Tensor,
        *,
        pos: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Sequence[torch.Tensor]:
        _reject_mask_kwargs(kwargs)
        inner = getattr(self.wrapped_encoder, "model", None)
        if not isinstance(inner, nn.Module):
            raise AdapterContractError(
                "all requires the official NeuralTrain REVE wrapper with a model attribute"
            )

        channel_indices = getattr(self.wrapped_encoder, "channel_indices", None)
        if channel_indices is not None:
            eeg = eeg[:, channel_indices]
        output = inner(eeg, pos=pos, return_output=True)
        if isinstance(output, torch.Tensor):
            raise AdapterContractError(
                "all requires return_output=True to expose the ordered layer sequence"
            )
        if not isinstance(output, (list, tuple)):
            raise AdapterContractError(
                "all requires return_output=True to return an ordered sequence"
            )
        return output


class UpstreamReveHeadModel(nn.Module):
    """Official NeuralBench model wrapper around an already-built NtReve."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        variant: str,
        n_outputs: int,
        dropout: float = DEFAULT_UPSTREAM_DROPOUT,
        query_token: torch.Tensor | None = None,
        query_initialization_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if variant == "last_tuned":
            validate_last_tuned_protocol(variant)
            if query_token is None:
                raise AdapterContractError("last_tuned requires an initialized cls_query_token")
        else:
            validate_upstream_head_variant(variant)
        self.encoder = encoder
        self.variant = variant
        self.all_layer_encoder = AllLayerReveEncoder(encoder) if variant == "all" else None
        # The upstream downstream classifier initializes its own query token;
        # the pretrained NtReve encoder has no classifier token in its state.
        # ``UpstreamReveHead`` performs that explicit torch.randn init after
        # the official experiment seed has been applied.
        embed_dim = _infer_embed_dim(encoder)
        if variant == "last_tuned":
            self.head = UpstreamReveHead(
                variant=variant,
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                dropout=dropout,
                query_token=query_token,
                query_initialization_metadata=query_initialization_metadata,
            )
        else:
            self.head = UpstreamReveHead(
                variant=variant,
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                dropout=dropout,
            )

    def _encode(
        self, eeg: torch.Tensor, channel_positions: torch.Tensor | None
    ) -> Any:
        if self.variant == "all":
            assert self.all_layer_encoder is not None
            return self.all_layer_encoder(eeg, pos=channel_positions)
        return self.encoder(eeg, pos=channel_positions)

    def forward(
        self,
        eeg: torch.Tensor,
        channel_positions: torch.Tensor | None = None,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        encoded = self._encode(eeg, channel_positions)
        if return_embedding:
            return self.head.pool_tokens(encoded)
        return self.head(encoded)


def _infer_embed_dim(encoder: nn.Module) -> int:
    for module in (encoder, _unwrap_reve_module(encoder)):
        value = getattr(module, "embed_dim", None)
        if isinstance(value, int):
            return value
    raise AdapterContractError("could not infer REVE embedding dimension")


def make_upstream_reve_wrapper(
    *,
    variant: str,
    dropout: float = DEFAULT_UPSTREAM_DROPOUT,
) -> Any:
    """Create a concrete official NeuralBench wrapper config lazily."""

    if variant == "last_tuned":
        validate_last_tuned_protocol(variant)
    else:
        validate_upstream_head_variant(variant)
    if dropout != DEFAULT_UPSTREAM_DROPOUT:
        raise ValueError(
            "the first upstream-head comparison fixes dropout=0.0; "
            f"got {dropout}"
        )

    from neuralbench.modules import DownstreamWrapper

    class UpstreamReveHeadWrapper(DownstreamWrapper):
        head_variant: str = variant
        head_dropout: float = float(dropout)

        def build(
            self,
            model: nn.Module,
            dummy_batch: dict[str, torch.Tensor | None],
            n_outputs: int,
            input_channel_names: list[str] | None = None,
        ) -> UpstreamReveHeadModel:
            if self.on_the_fly_preprocessor is not None:
                raise AdapterContractError("upstream REVE wrapper does not support preprocessing")
            if self.channel_adapter_config is not None:
                raise AdapterContractError("upstream REVE wrapper does not support channel adapters")
            if self.model_output_key is not None:
                raise AdapterContractError("upstream REVE wrapper requires model_output_key=None")
            if self.layers_to_freeze is not None or self.layers_to_unfreeze is not None:
                raise ProtocolMismatchError(
                    "upstream REVE head experiment requires full backbone fine-tuning"
                )
            if self.head_variant == "last_tuned":
                query_token, query_metadata = initialize_last_tuned_query(
                    model,
                    dummy_batch,
                    provenance=_NEURALBENCH_TRAIN_DUMMY_CONTEXT,
                )
                head_model = UpstreamReveHeadModel(
                    model,
                    variant=self.head_variant,
                    n_outputs=n_outputs,
                    dropout=self.head_dropout,
                    query_token=query_token,
                    query_initialization_metadata=query_metadata,
                )
                input_key = _encoder_primary_input_key(model)
                sample = dummy_batch[input_key]
                channel_positions = dummy_batch.get("channel_positions")
            else:
                head_model = UpstreamReveHeadModel(
                    model,
                    variant=self.head_variant,
                    n_outputs=n_outputs,
                    dropout=self.head_dropout,
                )
                input_name = next(iter(dummy_batch))
                sample = dummy_batch[input_name]
            if sample is None:
                raise AdapterContractError("dummy batch contains no EEG tensor")
            with torch.no_grad():
                head_model.eval()
                if self.head_variant == "last_tuned":
                    head_model(sample, channel_positions=channel_positions)
                else:
                    head_model(sample)
            head_model.train()
            return head_model

    return UpstreamReveHeadWrapper(
        model_output_key=None,
        aggregation=None,
        probe_config=None,
        head_variant=variant,
        head_dropout=float(dropout),
    )


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
    mismatches: list[str] = []
    for path, expected in experiment_checks.items():
        actual = _get_path(experiment, path)
        if actual is _MISSING or not _same(actual, expected):
            mismatches.append(f"{path}: expected {expected!r}, got {actual!r}")
    for path, expected in optimizer_checks.items():
        actual = _get_path(optimizer_config, path)
        if actual is _MISSING or not _same(actual, expected):
            mismatches.append(f"optimizer_config.{path}: expected {expected!r}, got {actual!r}")

    groups = _get_path(optimizer_config, "param_groups")
    if not isinstance(groups, (list, tuple)) or len(groups) != 2:
        mismatches.append("optimizer_config.param_groups must contain exactly base and query")
    else:
        expected_groups = (
            ("base", LAST_TUNED_BASE_LR),
            ("query", LAST_TUNED_QUERY_LR),
        )
        seen_parameter_names: set[str] = set()
        for index, (expected_name, expected_lr) in enumerate(expected_groups):
            group = groups[index]
            if not isinstance(group, Mapping) and not hasattr(group, "__dict__"):
                mismatches.append(
                    f"optimizer_config.param_groups[{index}] must be a mapping-like object"
                )
                continue
            group_path = f"optimizer_config.param_groups[{index}]"
            group_name = _get_path(group, "name")
            if group_name is _MISSING or group_name != expected_name:
                mismatches.append(
                    f"{group_path}.name: expected {expected_name!r}, got {group_name!r}"
                )
            group_lr = _get_path(group, "learning_rate")
            if group_lr is _MISSING or not _same(group_lr, expected_lr):
                mismatches.append(
                    f"{group_path}.learning_rate: expected {expected_lr!r}, got {group_lr!r}"
                )
            group_weight_decay = _get_path(group, "weight_decay")
            if group_weight_decay is _MISSING or not _same(
                group_weight_decay, LAST_TUNED_WEIGHT_DECAY
            ):
                mismatches.append(
                    f"{group_path}.weight_decay: expected {LAST_TUNED_WEIGHT_DECAY!r}, "
                    f"got {group_weight_decay!r}"
                )
            parameter_names = _get_path(group, "parameter_names")
            if (
                isinstance(parameter_names, str)
                or not isinstance(parameter_names, (list, tuple))
                or not all(isinstance(name, str) and name for name in parameter_names)
            ):
                mismatches.append(
                    f"{group_path}.parameter_names: expected a non-empty sequence of names, "
                    f"got {parameter_names!r}"
                )
                continue
            if len(parameter_names) != len(set(parameter_names)):
                mismatches.append(
                    f"{group_path}.parameter_names: duplicate parameter names"
                )
            overlap = seen_parameter_names.intersection(parameter_names)
            if overlap:
                mismatches.append(
                    "optimizer_config.param_groups.parameter_names: duplicate names "
                    f"{sorted(overlap)!r}"
                )
            seen_parameter_names.update(parameter_names)
            if expected_name == "base":
                if "head.query_token" in parameter_names:
                    mismatches.append(
                        f"{group_path}.parameter_names: base group must exclude "
                        "'head.query_token'"
                    )
                if "head.gate_logit" not in parameter_names:
                    mismatches.append(
                        f"{group_path}.parameter_names: base group must include "
                        "'head.gate_logit'"
                    )
            elif list(parameter_names) != ["head.query_token"]:
                mismatches.append(
                    f"{group_path}.parameter_names: query group must contain only "
                    "'head.query_token'"
                )

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
                mismatches.append(
                    f"downstream_model_wrapper.{field} must be None for full fine-tuning"
                )

    if n_total_params is not None and n_trainable_params != n_total_params:
        mismatches.append(
            "backbone trainability: expected all parameters trainable, "
            f"got {n_trainable_params}/{n_total_params}"
        )

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
            raise ValueError(
                f"upstream source hash mismatch for {relative}: "
                f"expected {expected}, got {digest}"
            )
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


__all__ = [
    "AdapterContractError",
    "AllLayerReveEncoder",
    "DEFAULT_UPSTREAM_DROPOUT",
    "HEAD_VARIANTS",
    "LAST_TUNED_HEAD_ARCHITECTURE",
    "LAST_TUNED_HEAD_SOURCE",
    "LAST_TUNED_INITIAL_ALPHA",
    "LAST_TUNED_BASE_LR",
    "LAST_TUNED_QUERY_LR",
    "LAST_TUNED_SCHEDULER_DIV_FACTOR",
    "LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR",
    "LAST_TUNED_SCHEDULER_MAX_LR",
    "LAST_TUNED_SCHEDULER_PCT_START",
    "LAST_TUNED_WEIGHT_DECAY",
    "LAST_TUNED_PROTOCOL_CLASS",
    "LAST_TUNED_PROTOCOL_VARIANTS",
    "OFFICIAL_HEAD_VARIANTS",
    "PROTOCOL_CONTRACT",
    "ProtocolMismatchError",
    "RMSNorm",
    "UPSTREAM_HEAD_HIDDEN_SIZE",
    "UPSTREAM_HEAD_INIT_CUTOFF",
    "UPSTREAM_HEAD_INIT_STD",
    "UPSTREAM_REVE_COMMIT",
    "UPSTREAM_REVE_FILE_HASHES",
    "UPSTREAM_REVE_REPOSITORY",
    "UPSTREAM_HEAD_VARIANTS",
    "UpstreamReveHead",
    "UpstreamReveHeadModel",
    "build_last_tuned_optimizer_config",
    "concatenate_all_layers",
    "initialize_last_tuned_query",
    "last_tuned_optimizer_metadata",
    "make_upstream_reve_wrapper",
    "runtime_metadata",
    "source_lock_metadata",
    "validate_head_variant",
    "validate_last_tuned_protocol",
    "validate_official_head_variant",
    "validate_official_protocol",
    "validate_upstream_head_variant",
    "verify_upstream_source_hashes",
]
