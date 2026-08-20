"""Train-dummy initialization and the fixed last_tuned optimizer protocol."""

from __future__ import annotations

import hashlib
import inspect
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:
    from .reve_contract import (
        AdapterContractError,
        LAST_TUNED_BASE_LR,
        LAST_TUNED_QUERY_LR,
        LAST_TUNED_SCHEDULER_DIV_FACTOR,
        LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
        LAST_TUNED_SCHEDULER_MAX_LR,
        LAST_TUNED_SCHEDULER_PCT_START,
        LAST_TUNED_WEIGHT_DECAY,
        ProtocolMismatchError,
    )
except ImportError:
    from reve_contract import (
        AdapterContractError,
        LAST_TUNED_BASE_LR,
        LAST_TUNED_QUERY_LR,
        LAST_TUNED_SCHEDULER_DIV_FACTOR,
        LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
        LAST_TUNED_SCHEDULER_MAX_LR,
        LAST_TUNED_SCHEDULER_PCT_START,
        LAST_TUNED_WEIGHT_DECAY,
        ProtocolMismatchError,
    )

_NEURALBENCH_TRAIN_DUMMY_CONTEXT = object()

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


# ---------------------------------------------------------------------------
# last_tuned query initialization and optimizer
# ---------------------------------------------------------------------------


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
                    raise AdapterContractError("last_tuned encoder must return a final-token tensor")
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
        metadata["query_initialization_subject_ids"] = _json_safe_value(dummy_batch["subject_ids"])
    return query, metadata



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
        raise ProtocolMismatchError("trainer.estimated_stepping_batches must be a positive integer")
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
        raise ProtocolMismatchError("last_tuned optimizer requires model.head.query_token")
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
        raise ProtocolMismatchError("last_tuned base optimizer group must contain trainable parameters")

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


def _validate_tuned_optimizer_group(
    index: int,
    group: Mapping[str, Any],
    expected_parameters: Sequence[nn.Parameter],
    *,
    max_lr: float,
    initial_lr: float,
) -> list[int]:
    """Validate one tuned optimizer group and return its parameter identities."""

    observed_parameters = list(group["params"])
    if [id(parameter) for parameter in observed_parameters] != [
        id(parameter) for parameter in expected_parameters
    ]:
        raise ProtocolMismatchError(
            f"last_tuned optimizer group {index} parameter membership changed"
        )

    expected_values = {
        "weight_decay": LAST_TUNED_WEIGHT_DECAY,
        "initial_lr": initial_lr,
        "max_lr": max_lr,
        "min_lr": initial_lr / LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
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
    return [id(parameter) for parameter in observed_parameters]


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
    observed_identities: list[int] = []
    for index, (group, parameters, max_lr, initial_lr) in enumerate(
        zip(
            optimizer.param_groups,
            expected_parameters,
            expected_max_lrs,
            expected_initial_lrs,
        )
    ):
        observed_identities.extend(
            _validate_tuned_optimizer_group(
                index,
                group,
                parameters,
                max_lr=max_lr,
                initial_lr=initial_lr,
            )
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
