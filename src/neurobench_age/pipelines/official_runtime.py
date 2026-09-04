"""Scoped NeuralBench patch lifecycle for the official REVE runner.

This module owns temporary monkeypatches and cleanup. It receives the facade as
the hooks object so tests and callers can keep replacing the public seams.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import types
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from ..core.evidence import add_declared_head_bucket, parameter_buckets_from_model

_CONFIGURE_OPTIMIZERS_ABSENT = object()
_TRAINING_STEP_ABSENT = object()
H6_GATE_STABILITY_LAMBDA = 0.001
H6_NOISE_SCALE = 0.01
AUGMENTATION_CONSISTENCY_LAMBDA = 0.05
AUGMENTATION_CONSISTENCY_NOISE_SCALE = 0.01
AUGMENTATION_CONSISTENCY_BATCH_SIZE = 8


class CorrelationAuxiliaryLoss(nn.Module):
    """Keep MSE primary while adding a bounded batch Pearson objective."""

    def __init__(self, base_loss: nn.Module, *, coefficient: float = 0.02) -> None:
        super().__init__()
        if not isinstance(base_loss, nn.Module):
            raise TypeError("correlation auxiliary loss requires an nn.Module base loss")
        if not math.isfinite(float(coefficient)) or float(coefficient) < 0.0:
            raise ValueError("correlation coefficient must be finite and non-negative")
        self.base_loss = base_loss
        self.coefficient = float(coefficient)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = self.base_loss(prediction, target)
        prediction_values = prediction.reshape(prediction.shape[0], -1).mean(dim=1)
        target_values = target.reshape(target.shape[0], -1).mean(dim=1)
        if prediction_values.shape[0] < 2:
            auxiliary = prediction.new_zeros(())
        else:
            prediction_centered = prediction_values - prediction_values.mean()
            target_centered = target_values - target_values.mean()
            denominator = torch.sqrt(
                prediction_centered.square().sum() * target_centered.square().sum()
            )
            if not torch.isfinite(denominator) or float(denominator.detach()) <= 1e-8:
                auxiliary = prediction.new_zeros(())
            else:
                correlation = (
                    prediction_centered * target_centered
                ).sum() / denominator
                auxiliary = float(self.coefficient) * (1.0 - correlation.clamp(-1.0, 1.0))
        # NeuralBench expects its configured loss callable to return one scalar
        # tensor.  Keep the components as attributes for optional diagnostics,
        # while returning the scalar consumed by Lightning's training step.
        self.last_mse = mse.detach()
        self.last_pearson_aux = auxiliary.detach()
        return mse + auxiliary


class TrainingOnlyTargetZScore:
    """Small scaler fitted exclusively through training batches."""

    def __init__(self) -> None:
        self._mean: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None
        self._n_samples_seen = 0

    def partial_fit(self, values: torch.Tensor, *, split: str = "train") -> "TrainingOnlyTargetZScore":
        if split != "train":
            raise ValueError("target z-score accepts training targets only")
        values = torch.as_tensor(values).detach()
        flat = values.reshape(-1, 1).float()
        if flat.numel() == 0:
            raise ValueError("target z-score cannot fit an empty batch")
        count = flat.shape[0]
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        if self._mean is None:
            self._mean = batch_mean
            self._scale = batch_var.sqrt().clamp_min(1e-6)
            self._n_samples_seen = count
            return self
        previous_count = self._n_samples_seen
        previous_mean = self._mean
        previous_var = self._scale.square()
        total = previous_count + count
        mean = (previous_count * previous_mean + count * batch_mean) / total
        var = (
            previous_count * previous_var
            + count * batch_var
            + previous_count * count / total * (previous_mean - batch_mean).square()
        ) / total
        self._mean = mean
        self._scale = var.sqrt().clamp_min(1e-6)
        self._n_samples_seen = total
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._scale is None:
            raise RuntimeError("target z-score is not fitted")
        return (values - self._mean.to(values.device, values.dtype)) / self._scale.to(values.device, values.dtype)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._scale is None:
            raise RuntimeError("target z-score is not fitted")
        return values * self._scale.to(values.device, values.dtype) + self._mean.to(values.device, values.dtype)

    def statistics_hash(self) -> str:
        if self._mean is None or self._scale is None:
            return hashlib.sha256(b"unfitted").hexdigest()
        payload = self._mean.detach().cpu().contiguous().numpy().tobytes() + self._scale.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()


def target_scaler_metadata(
    *,
    scaler: TrainingOnlyTargetZScore,
    train_subject_ids: Sequence[str],
    train_timeline_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "zscore",
        "fit_split": "train",
        "n_samples_seen": int(scaler._n_samples_seen),
        "mean": None if scaler._mean is None else float(scaler._mean.flatten()[0]),
        "scale": None if scaler._scale is None else float(scaler._scale.flatten()[0]),
        "statistics_sha256": scaler.statistics_hash(),
        "train_subject_ids": sorted(set(str(item) for item in train_subject_ids)),
        "train_timeline_ids": sorted(set(str(item) for item in train_timeline_ids)),
        "validation_subject_ids": [],
        "validation_timeline_ids": [],
        "test_subject_ids": [],
        "test_timeline_ids": [],
    }


def build_training_loss(robust_loss: str = "mse") -> nn.Module:
    if robust_loss == "mse":
        return nn.MSELoss()
    if robust_loss == "smooth_l1":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError("robust_loss must be 'mse' or 'smooth_l1'")


def _h6_noise_seed(*, run_seed: int, epoch: int, batch_idx: int) -> int:
    """Derive a stable, private seed for one training-only paired view."""

    return int(run_seed) + 1009 * int(epoch) + int(batch_idx)


def make_h6_training_view(
    neuro: Any,
    *,
    run_seed: int,
    epoch: int,
    batch_idx: int,
    noise_scale: float = H6_NOISE_SCALE,
) -> Any:
    """Create the bounded-noise H6 view without touching global RNG state."""

    import torch

    if not isinstance(neuro, torch.Tensor) or neuro.ndim < 2:
        raise TypeError("H6 expects a tensor with batch and time dimensions")
    if not math.isfinite(float(noise_scale)) or noise_scale < 0.0:
        raise ValueError("H6 noise_scale must be finite and non-negative")
    generator = torch.Generator(device=neuro.device)
    generator.manual_seed(_h6_noise_seed(run_seed=run_seed, epoch=epoch, batch_idx=batch_idx))
    noise = torch.randn(
        neuro.shape,
        generator=generator,
        device=neuro.device,
        dtype=neuro.dtype,
    ).clamp(-3.0, 3.0)
    scale = torch.nan_to_num(
        neuro.detach().std(dim=-1, keepdim=True, unbiased=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(1e-6)
    return neuro + float(noise_scale) * scale * noise


def _augmentation_consistency_noise_seed(*, run_seed: int, epoch: int, batch_idx: int) -> int:
    """Derive a private deterministic seed for a paired training view."""

    return 1_000_003 + int(run_seed) + 1009 * int(epoch) + int(batch_idx)


def make_augmentation_consistency_view(
    neuro: Any,
    *,
    run_seed: int,
    epoch: int,
    batch_idx: int,
    noise_scale: float = AUGMENTATION_CONSISTENCY_NOISE_SCALE,
) -> Any:
    """Jitter each representation token independently without global RNG state.

    ``neuro`` is the standardized EEG input tensor consumed by the REVE
    wrapper. The returned tensor has the same shape: every input sample
    receives independent bounded noise across its final time dimension, while
    no information is mixed between examples.
    """

    if not isinstance(neuro, torch.Tensor) or neuro.ndim < 2:
        raise TypeError("augmentation consistency expects a tensor with batch and token dimensions")
    if not torch.is_floating_point(neuro):
        raise TypeError("augmentation consistency expects a floating-point representation")
    if not math.isfinite(float(noise_scale)) or noise_scale < 0.0:
        raise ValueError("augmentation consistency noise_scale must be finite and non-negative")
    generator = torch.Generator(device=neuro.device)
    generator.manual_seed(
        _augmentation_consistency_noise_seed(
            run_seed=run_seed,
            epoch=epoch,
            batch_idx=batch_idx,
        )
    )
    noise = torch.randn(
        neuro.shape,
        generator=generator,
        device=neuro.device,
        dtype=neuro.dtype,
    ).clamp(-3.0, 3.0)
    token_scale = torch.nan_to_num(
        neuro.detach().std(dim=-1, keepdim=True, unbiased=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(1e-6)
    return neuro + float(noise_scale) * token_scale * noise


def augmentation_consistency_loss(
    prediction_one: Any,
    prediction_two: Any,
    *,
    lambda_consistency: float = AUGMENTATION_CONSISTENCY_LAMBDA,
) -> tuple[Any, Any]:
    """Return weighted and raw MSE agreement losses for paired predictions."""

    if not isinstance(prediction_one, torch.Tensor) or not isinstance(prediction_two, torch.Tensor):
        raise TypeError("augmentation consistency predictions must be tensors")
    if prediction_one.shape != prediction_two.shape:
        raise ValueError("augmentation consistency predictions must have identical shapes")
    if not math.isfinite(float(lambda_consistency)) or lambda_consistency < 0.0:
        raise ValueError("augmentation consistency lambda must be finite and non-negative")
    if not torch.isfinite(prediction_one).all() or not torch.isfinite(prediction_two).all():
        raise RuntimeError("augmentation consistency received non-finite predictions")
    raw = torch.nn.functional.mse_loss(prediction_two, prediction_one.detach())
    if not torch.isfinite(raw):
        raise RuntimeError("augmentation consistency loss is non-finite")
    return float(lambda_consistency) * raw, raw


def _batch_prefix(data: Mapping[str, Any], *, size: int, batch_size: int) -> dict[str, Any]:
    """Slice fields aligned to the first batch dimension for a small paired view."""

    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            result[key] = value[:size]
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            result[key] = value[:size]
        else:
            result[key] = value
    return result


def h6_gate_consistency_loss(
    alpha_one: Any,
    alpha_two: Any,
    *,
    lambda_gate: float = H6_GATE_STABILITY_LAMBDA,
) -> Any:
    """Return the training-only weighted squared gate disagreement."""

    import torch

    if not isinstance(alpha_one, torch.Tensor) or not isinstance(alpha_two, torch.Tensor):
        raise TypeError("H6 gate values must be tensors")
    if alpha_one.shape != alpha_two.shape:
        raise ValueError("H6 gate views must have identical shapes")
    if not math.isfinite(float(lambda_gate)) or lambda_gate < 0.0:
        raise ValueError("H6 lambda_gate must be finite and non-negative")
    if not torch.isfinite(alpha_one).all() or not torch.isfinite(alpha_two).all():
        raise RuntimeError("H6 gate consistency received non-finite values")
    return float(lambda_gate) * (alpha_one - alpha_two).pow(2).mean()


def _find_reliability_head(brain_module: Any) -> Any:
    """Find the shared H2/H6 head inside a prepared BrainModule."""

    for module in brain_module.modules():
        if module.__class__.__name__ == "MeanReliabilityShrinkageHead":
            return module
    raise RuntimeError("H6 model did not expose MeanReliabilityShrinkageHead")


def _patch_h6_training_step(
    brain_module: Any,
    *,
    run_seed: int,
    patched_modules: list[dict[str, Any]],
) -> None:
    """Add H6's paired-view loss to one BrainModule instance."""

    if brain_module is None or not isinstance(patched_modules, list):
        raise TypeError("H6 requires a BrainModule and a patch record list")
    if any(record["module"] is brain_module for record in patched_modules):
        raise RuntimeError("H6 BrainModule was already patched in this run")
    attributes = getattr(brain_module, "__dict__", None)
    if not isinstance(attributes, dict):
        raise TypeError("H6 BrainModule must expose instance attributes")
    previous = attributes.get("training_step", _TRAINING_STEP_ABSENT)
    original = brain_module.training_step
    record = {"module": brain_module, "previous": previous}
    patched_modules.append(record)

    def training_step(module: Any, batch: Any, batch_idx: int) -> Any:
        import torch
        from types import SimpleNamespace

        base_loss = original(batch, batch_idx)
        head = _find_reliability_head(module)
        alpha_one = getattr(head, "_last_gate_values", None)
        if not isinstance(alpha_one, torch.Tensor):
            raise RuntimeError("H6 original training view did not produce gate values")
        alpha_one_for_audit = alpha_one.detach().clone()
        data = getattr(batch, "data", None)
        if not isinstance(data, Mapping) or "neuro" not in data:
            raise RuntimeError("H6 training batch does not expose data['neuro']")
        paired_data = dict(data)
        paired_data["neuro"] = make_h6_training_view(
            data["neuro"],
            run_seed=run_seed,
            epoch=int(getattr(module, "current_epoch", 0)) + 1,
            batch_idx=batch_idx,
        )
        module.model_forward(SimpleNamespace(data=paired_data))
        alpha_two = getattr(head, "_last_gate_values", None)
        if not isinstance(alpha_two, torch.Tensor):
            raise RuntimeError("H6 paired training view did not produce gate values")
        consistency_loss = h6_gate_consistency_loss(alpha_one, alpha_two)
        raw_consistency = (alpha_one - alpha_two).pow(2).mean()
        if not torch.isfinite(raw_consistency):
            raise RuntimeError("H6 gate consistency is non-finite")
        module._h6_last_gate_consistency = raw_consistency.detach()
        module.log(
            "train/gate_consistency",
            raw_consistency,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=alpha_one.shape[0],
            sync_dist=getattr(module.trainer, "world_size", 1) > 1,
        )
        # ReliabilityGateAudit must observe the unperturbed first view.
        head._last_gate_values = alpha_one_for_audit
        return base_loss + consistency_loss

    try:
        brain_module.training_step = types.MethodType(training_step, brain_module)
    except BaseException:
        patched_modules.pop()
        raise


def _restore_h6_training_steps(patched_modules: list[dict[str, Any]]) -> None:
    """Restore all instance-level H6 training-step patches."""

    errors: list[BaseException] = []
    while patched_modules:
        record = patched_modules.pop()
        module = record["module"]
        previous = record["previous"]
        try:
            if previous is _TRAINING_STEP_ABSENT:
                attributes = getattr(module, "__dict__", {})
                if "training_step" in attributes:
                    delattr(module, "training_step")
            else:
                module.training_step = previous
        except BaseException as error:
            errors.append(error)
    if errors:
        error = RuntimeError("failed to restore one or more H6 training-step patches")
        for restoration_error in errors:
            error.add_note(repr(restoration_error))
        raise error


def _patch_augmentation_consistency_training_step(
    brain_module: Any,
    *,
    run_seed: int,
    lambda_consistency: float = AUGMENTATION_CONSISTENCY_LAMBDA,
    noise_scale: float = AUGMENTATION_CONSISTENCY_NOISE_SCALE,
    consistency_batch_size: int = AUGMENTATION_CONSISTENCY_BATCH_SIZE,
    patched_modules: list[dict[str, Any]],
) -> None:
    """Add a train-only paired-view prediction agreement loss."""

    if brain_module is None or not isinstance(patched_modules, list):
        raise TypeError("augmentation consistency requires a BrainModule and a patch record list")
    if not math.isfinite(float(lambda_consistency)) or lambda_consistency < 0.0:
        raise ValueError("augmentation consistency lambda must be finite and non-negative")
    if not math.isfinite(float(noise_scale)) or noise_scale < 0.0:
        raise ValueError("augmentation consistency noise_scale must be finite and non-negative")
    if isinstance(consistency_batch_size, bool) or not isinstance(consistency_batch_size, int) or consistency_batch_size < 1:
        raise ValueError("augmentation consistency batch size must be a positive integer")
    if any(record["module"] is brain_module for record in patched_modules):
        raise RuntimeError("augmentation consistency BrainModule was already patched in this run")
    attributes = getattr(brain_module, "__dict__", None)
    if not isinstance(attributes, dict):
        raise TypeError("augmentation consistency BrainModule must expose instance attributes")
    previous = attributes.get("training_step", _TRAINING_STEP_ABSENT)
    original = brain_module.training_step
    record = {"module": brain_module, "previous": previous}
    patched_modules.append(record)

    def training_step(module: Any, batch: Any, batch_idx: int) -> Any:
        from types import SimpleNamespace

        data = getattr(batch, "data", None)
        if not isinstance(data, Mapping) or "neuro" not in data:
            raise RuntimeError("augmentation consistency batch does not expose data['neuro']")
        neuro = data["neuro"]
        if not isinstance(neuro, torch.Tensor) or neuro.ndim < 1:
            raise TypeError("augmentation consistency batch data['neuro'] must be batched")
        batch_size = int(neuro.shape[0])
        pair_size = min(int(consistency_batch_size), batch_size)
        captured: dict[str, torch.Tensor] = {}
        model_forward_attributes = getattr(module, "__dict__", None)
        if not isinstance(model_forward_attributes, dict):
            raise TypeError("augmentation consistency BrainModule must expose instance attributes")
        previous_model_forward = model_forward_attributes.get("model_forward", _TRAINING_STEP_ABSENT)
        original_model_forward = module.model_forward

        def capture_model_forward(model: Any, clean_batch: Any) -> Any:
            prediction = original_model_forward(clean_batch)
            if not isinstance(prediction, torch.Tensor):
                raise TypeError("augmentation consistency model_forward must return a tensor")
            captured["clean_prediction"] = prediction
            return prediction

        try:
            module.model_forward = types.MethodType(capture_model_forward, module)
            base_loss = original(batch, batch_idx)
        finally:
            if previous_model_forward is _TRAINING_STEP_ABSENT:
                if "model_forward" in model_forward_attributes:
                    delattr(module, "model_forward")
            else:
                module.model_forward = previous_model_forward
        clean_prediction = captured.get("clean_prediction")
        if clean_prediction is None:
            raise RuntimeError("augmentation consistency did not capture the clean prediction")
        paired_data = _batch_prefix(data, size=pair_size, batch_size=batch_size)
        paired_data["neuro"] = make_augmentation_consistency_view(
            neuro[:pair_size],
            run_seed=run_seed,
            epoch=int(getattr(module, "current_epoch", 0)) + 1,
            batch_idx=batch_idx,
            noise_scale=noise_scale,
        )
        augmented_prediction = module.model_forward(SimpleNamespace(data=paired_data))
        consistency_loss, raw_consistency = augmentation_consistency_loss(
            clean_prediction[:pair_size],
            augmented_prediction,
            lambda_consistency=lambda_consistency,
        )
        module._augmentation_consistency_last_loss = raw_consistency.detach()
        trainer = getattr(module, "trainer", None)
        module.log(
            "train/augmentation_consistency",
            raw_consistency,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            batch_size=pair_size,
            sync_dist=getattr(trainer, "world_size", 1) > 1,
        )
        return base_loss + consistency_loss

    try:
        brain_module.training_step = types.MethodType(training_step, brain_module)
    except BaseException:
        patched_modules.pop()
        raise


def _restore_augmentation_consistency_training_steps(
    patched_modules: list[dict[str, Any]],
) -> None:
    """Restore all instance-level augmentation consistency patches."""

    errors: list[BaseException] = []
    while patched_modules:
        record = patched_modules.pop()
        module = record["module"]
        previous = record["previous"]
        try:
            if previous is _TRAINING_STEP_ABSENT:
                attributes = getattr(module, "__dict__", {})
                if "training_step" in attributes:
                    delattr(module, "training_step")
            else:
                module.training_step = previous
        except BaseException as error:
            errors.append(error)
    if errors:
        error = RuntimeError(
            "failed to restore one or more augmentation consistency training-step patches"
        )
        for restoration_error in errors:
            error.add_note(repr(restoration_error))
        raise error


def _last_tuned_configure_optimizers(brain_module: Any, *, hooks: Any) -> dict[str, Any]:
    """Build the tuned optimizer from one prepared BrainModule instance."""

    model = getattr(brain_module, "model", None)
    trainer = getattr(brain_module, "trainer", None)
    return hooks._load_reve_helpers().build_last_tuned_optimizer_config(model, trainer=trainer)


def _patch_last_tuned_configure_optimizers(
    brain_module: Any,
    patched_modules: list[dict[str, Any]],
    *,
    hooks: Any,
) -> None:
    """Install one instance-only optimizer override and record restoration state."""

    if brain_module is None:
        raise RuntimeError("last_tuned prepare_pl_module created no _brain_module")
    if not isinstance(patched_modules, list):
        raise TypeError("patched_modules must be a per-run list")
    if any(record["module"] is brain_module for record in patched_modules):
        raise RuntimeError("last_tuned BrainModule was already patched in this run")

    instance_attributes = getattr(brain_module, "__dict__", None)
    if not isinstance(instance_attributes, dict):
        raise TypeError("last_tuned BrainModule must expose instance attributes")
    previous = instance_attributes.get("configure_optimizers", _CONFIGURE_OPTIMIZERS_ABSENT)
    record = {
        "module": brain_module,
        "previous": previous,
    }
    patched_modules.append(record)
    try:
        def configure_optimizers(module: Any) -> dict[str, Any]:
            return _last_tuned_configure_optimizers(module, hooks=hooks)

        brain_module.configure_optimizers = types.MethodType(configure_optimizers, brain_module)
    except BaseException:
        patched_modules.pop()
        raise


def _restore_last_tuned_configure_optimizers(
    patched_modules: list[dict[str, Any]],
) -> None:
    """Restore every patched instance in reverse installation order."""

    restoration_errors: list[BaseException] = []
    while patched_modules:
        record = patched_modules.pop()
        brain_module = record["module"]
        previous = record["previous"]
        try:
            if previous is _CONFIGURE_OPTIMIZERS_ABSENT:
                instance_attributes = getattr(brain_module, "__dict__", {})
                if "configure_optimizers" in instance_attributes:
                    delattr(brain_module, "configure_optimizers")
            else:
                brain_module.configure_optimizers = previous
        except BaseException as error:
            restoration_errors.append(error)
    if restoration_errors:
        error = RuntimeError("failed to restore one or more last_tuned configure_optimizers patches")
        for restoration_error in restoration_errors:
            error.add_note(repr(restoration_error))
        raise error


def _install_selective_eeglab_mat_reader() -> dict[str, Any]:
    """Use SciPy's cell simplifier for task-only EEGLAB acquisition.

    Some HBN ``.set`` files contain nested MATLAB cell structures that MNE's
    default recursive converter compares as arrays, which raises an ambiguous
    truth-value error.  ``simplify_cells=True`` produces the same logical
    MATLAB content in a form MNE can consume.  Keep this process-local and
    scoped to one selective benchmark run.
    """

    import scipy.io
    import mne.io.eeglab.eeglab as eeglab

    original_readmat = eeglab._readmat

    def readmat(
        fname: Any,
        uint16_codec: str | None = None,
        *,
        preload: bool = False,
    ) -> Any:
        return scipy.io.loadmat(
            fname,
            struct_as_record=False,
            squeeze_me=True,
            simplify_cells=True,
            uint16_codec=uint16_codec,
        )

    eeglab._readmat = readmat
    return {"module": eeglab, "readmat": original_readmat}


def _should_use_simplified_eeglab_reader(data_mode: str) -> bool:
    """Use the HBN EEGLAB compatibility reader for manifest-backed data too."""

    return data_mode in {"manifest", "selective_task"}


def _find_nested_study(step: Any, study_class: type[Any]) -> Any | None:
    """Find one concrete study inside a NeuralSet Chain-like pipeline."""

    seen: set[int] = set()

    def visit(node: Any) -> Any | None:
        if isinstance(node, study_class):
            return node
        node_id = id(node)
        if node_id in seen:
            return None
        seen.add(node_id)
        children = getattr(node, "steps", None)
        if isinstance(children, Mapping):
            children = list(children.values())
        if not isinstance(children, (list, tuple)):
            return None
        for child in children:
            found = visit(child)
            if found is not None:
                return found
        return None

    return visit(step)


def _last_tuned_report_metadata(
    *,
    query_metadata: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten resolved tuning state into the stable report representation."""

    scheduler = optimizer_config["scheduler"]
    scheduler_kwargs = scheduler["kwargs"]
    return {
        **dict(query_metadata),
        "optimizer": optimizer_config["optimizer"]["name"],
        "base_learning_rate": optimizer_config["optimizer"]["lr"],
        "query_learning_rate": optimizer_config["param_groups"][1]["learning_rate"],
        "weight_decay": optimizer_config["optimizer"]["kwargs"]["weight_decay"],
        "scheduler": scheduler["name"],
        "scheduler_max_lr": list(scheduler_kwargs["max_lr"]),
        "scheduler_pct_start": scheduler_kwargs["pct_start"],
        "scheduler_anneal_strategy": scheduler_kwargs["anneal_strategy"],
        "scheduler_div_factor": scheduler_kwargs["div_factor"],
        "scheduler_final_div_factor": scheduler_kwargs["final_div_factor"],
        "scheduler_interval": scheduler["interval"],
        "scheduler_frequency": scheduler["frequency"],
        "optimizer_param_groups": list(optimizer_config["param_groups"]),
        "monitor": "val/pearsonr",
        "checkpoint_selection_monitor": "val/pearsonr",
        "test_pearsonr_role": "diagnostic_only",
    }


def _merge_last_tuned_result_metadata(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the resolved tuning metadata captured from the official test pass."""

    metadata: dict[str, Any] = {}
    for result in results:
        candidate = result.get("tuning_metadata")
        if isinstance(candidate, Mapping):
            metadata.update(candidate)
    return metadata


def _capture_test_result(
    result: Mapping[str, Any],
    *,
    head_variant: str,
    experiment_id: int,
    tuning_metadata_by_experiment: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Copy one official result and attach late-bound tuned metadata."""

    captured = dict(result)
    if head_variant != "last_tuned":
        return captured
    metadata = captured.get("tuning_metadata")
    merged = dict(metadata) if isinstance(metadata, Mapping) else {}
    merged.update(tuning_metadata_by_experiment.get(experiment_id, {}))
    captured["tuning_metadata"] = merged
    return captured


def _read_validation_history(path: Path, *, seed: int) -> list[dict[str, Any]]:
    """Read and validate strict one-based validation history records."""

    if not path.is_file():
        raise RuntimeError(f"strict validation history is missing: {path}")

    records: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid strict validation JSON at line {line_number}") from error
        if not isinstance(row, Mapping):
            raise RuntimeError(f"strict validation record at line {line_number} is not an object")
        row_seed = row.get("seed")
        epoch = row.get("epoch")
        metric = row.get("val/pearsonr")
        if row_seed != seed or isinstance(row_seed, bool):
            raise RuntimeError(f"strict validation seed mismatch at line {line_number}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise RuntimeError(f"strict validation epoch is invalid at line {line_number}")
        if epoch in seen_epochs:
            raise RuntimeError(f"duplicate epoch in strict validation history: {epoch}")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise RuntimeError(f"strict validation val/pearsonr is invalid at line {line_number}")
        metric_value = float(metric)
        if not math.isfinite(metric_value):
            raise RuntimeError(f"strict validation val/pearsonr is non-finite at line {line_number}")
        seen_epochs.add(epoch)
        records.append({"seed": seed, "epoch": epoch, "val/pearsonr": metric_value})

    if not records:
        raise RuntimeError(f"strict validation history is empty: {path}")
    return records


def _resolve_selected_validation(
    checkpoint_path: Path,
    validation_history_path: Path,
    *,
    seed: int,
) -> dict[str, int | float]:
    """Bind the checkpoint's raw epoch to exactly one validation record."""

    import torch

    if not checkpoint_path.is_file():
        raise RuntimeError(f"strict selected checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("strict checkpoint payload is not a mapping")
    raw_epoch = checkpoint.get("epoch")
    if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 0:
        raise RuntimeError("strict checkpoint does not contain a valid integer epoch")

    selected_epoch = raw_epoch + 1
    records = _read_validation_history(validation_history_path, seed=seed)
    matches = [row for row in records if row["epoch"] == selected_epoch]
    if len(matches) != 1:
        raise RuntimeError("strict selected epoch must match exactly one validation record")
    return {
        "checkpoint_epoch_zero_based": raw_epoch,
        "selected_epoch": selected_epoch,
        "selected_val_pearsonr": matches[0]["val/pearsonr"],
    }


def _freeze_provenance_snapshot(source_path: Path) -> Path:
    """Create an immutable per-run snapshot of a mutable upstream artifact."""

    if not source_path.is_file():
        raise RuntimeError(f"strict provenance file is missing: {source_path}")
    snapshot_path = source_path.with_name("strict_provenance_config.yaml")
    data = source_path.read_bytes()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.is_file():
        if snapshot_path.read_bytes() != data:
            raise RuntimeError(f"strict provenance snapshot differs: {snapshot_path}")
        return snapshot_path

    temporary_path = snapshot_path.with_name(
        f".{snapshot_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, snapshot_path)
            _fsync_directory(snapshot_path.parent)
        except FileExistsError as error:
            if not snapshot_path.is_file() or snapshot_path.read_bytes() != data:
                raise RuntimeError(f"strict provenance snapshot differs: {snapshot_path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return snapshot_path


def _build_strict_selection_record(
    *,
    checkpoint_path: Path,
    official_config_path: Path,
    manifest_path: Path | None = None,
    provenance_path: Path | None = None,
    acquisition_provenance_path: Path | None = None,
    acquisition_provenance_sha256: str | None = None,
    data_mode: str = "manifest",
    timeline_count: int | None = None,
    validation_history_path: Path,
    selection_monitor: str,
    selection_mode: str,
    seed: int,
    head_variant: str,
    strict_final_test: bool,
    sha256_file: Any,
) -> dict[str, Any]:
    """Build the immutable provenance record without writing it."""

    if selection_monitor != "val/pearsonr" or selection_mode != "max":
        raise RuntimeError("strict selection must monitor val/pearsonr in max mode")
    if data_mode not in {"manifest", "full", "selective_task"}:
        raise RuntimeError(f"unsupported strict data mode: {data_mode!r}")
    selected = _resolve_selected_validation(checkpoint_path, validation_history_path, seed=seed)
    immutable_config_path = _freeze_provenance_snapshot(official_config_path)
    if not immutable_config_path.is_file():
        raise RuntimeError(f"strict provenance file is missing: {immutable_config_path}")

    if data_mode == "manifest":
        if manifest_path is None:
            raise RuntimeError("manifest strict selection requires manifest_path")
        if provenance_path is None:
            provenance_path = manifest_path
        if not manifest_path.is_file():
            raise RuntimeError(f"strict provenance file is missing: {manifest_path}")
    else:
        if manifest_path is not None:
            raise RuntimeError("non-manifest strict selection must not include manifest_path")
        if provenance_path is None:
            raise RuntimeError("non-manifest strict selection requires provenance_path")
        if isinstance(timeline_count, bool) or not isinstance(timeline_count, int) or timeline_count < 1:
            raise RuntimeError("non-manifest strict selection requires a positive timeline_count")
        if data_mode == "selective_task":
            if acquisition_provenance_path is None or acquisition_provenance_sha256 is None:
                raise RuntimeError("selective strict selection requires acquisition provenance")
            if not acquisition_provenance_path.is_file():
                raise RuntimeError(
                    f"selective acquisition provenance is missing: {acquisition_provenance_path}"
                )
            actual_acquisition_digest = sha256_file(acquisition_provenance_path)
            if actual_acquisition_digest != acquisition_provenance_sha256:
                raise RuntimeError("selective acquisition provenance hash changed")
            sidecar = acquisition_provenance_path.with_suffix(".sha256")
            if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != actual_acquisition_digest:
                raise RuntimeError("selective acquisition provenance sidecar mismatch")
    if not provenance_path.is_file():
        raise RuntimeError(f"strict provenance file is missing: {provenance_path}")

    manifest_sha256 = sha256_file(manifest_path) if manifest_path is not None else None
    return {
        "schema_version": "1.0",
        "evaluation_protocol": "strict",
        "data_mode": data_mode,
        "timeline_count": timeline_count,
        "selection_monitor": selection_monitor,
        "selection_mode": selection_mode,
        "selection_rule": {
            "primary": "max_validation_pearsonr",
            "tie_breaking": "official_checkpoint_callback",
        },
        "seed": int(seed),
        "head_variant": head_variant,
        **selected,
        "strict_final_test": bool(strict_final_test),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_config_path": str(immutable_config_path.resolve()),
        "official_config_sha256": sha256_file(immutable_config_path),
        "manifest_path": str(manifest_path.resolve()) if manifest_path is not None else None,
        "manifest_sha256": manifest_sha256,
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": sha256_file(provenance_path),
        "acquisition_provenance_path": (
            str(acquisition_provenance_path.resolve())
            if acquisition_provenance_path is not None
            else None
        ),
        "acquisition_provenance_sha256": (
            sha256_file(acquisition_provenance_path)
            if acquisition_provenance_path is not None
            else None
        ),
        "validation_history_path": str(validation_history_path.resolve()),
        "validation_history_sha256": sha256_file(validation_history_path),
        "test_status": "sealed" if strict_final_test else "withheld",
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_bytes_atomic(
    path: Path,
    data: bytes,
    *,
    remove_final_on_failure: bool = False,
) -> None:
    """Replace one file atomically and clean up incomplete evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        if remove_final_on_failure:
            path.unlink(missing_ok=True)
        raise


def _replace_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON object used as late-bound run metadata."""

    data = (json.dumps(payload, indent=2, default=str) + "\n").encode("utf-8")
    _replace_bytes_atomic(path, data)


def _run_metadata_paths(experiment: Any, canonical_run_dir: Path) -> list[Path]:
    """Return official and canonical destinations for late-bound metadata.

    NeuralBench can omit its internal UID directory for a later seed in a
    multi-seed process.  The canonical seed directory must therefore always
    receive a copy of the metadata used by the article-ready evidence audit.
    """

    paths: list[Path] = []
    infra = getattr(experiment, "infra", None)
    uid_folder = getattr(infra, "uid_folder", None)
    if callable(uid_folder):
        uid_path = uid_folder()
        if uid_path is not None:
            paths.append(Path(uid_path) / "run_metadata.json")
    canonical_path = Path(canonical_run_dir) / "run_metadata.json"
    if canonical_path not in paths:
        paths.append(canonical_path)
    return paths


def _publish_json_create_if_absent(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON atomically without replacing an existing evidence file."""

    data = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
            _fsync_directory(path.parent)
        except FileExistsError as error:
            if not path.is_file() or path.read_bytes() != data:
                raise RuntimeError(f"existing strict evidence differs: {path}") from error
            return
    finally:
        temp_path.unlink(missing_ok=True)


def _create_test_start_marker(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a durable, exclusive marker before consuming test data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(payload)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"strict test marker already exists: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        # The marker intentionally remains when a write is interrupted: its
        # existence conservatively consumes the one permitted test access.
        raise


def _create_test_completed_marker(path: Path, payload: Mapping[str, Any]) -> None:
    """Record successful post-test integrity verification exactly once."""

    _publish_json_create_if_absent(path, payload)


def _extract_official_test_pearson(results: Sequence[Mapping[str, Any]]) -> float:
    """Extract only the pinned official ``test/pearsonr`` result."""

    if len(results) != 1:
        raise RuntimeError("strict final test requires exactly one official result")
    result = results[0]
    if "test/pearsonr" not in result:
        raise RuntimeError("official strict result is missing test/pearsonr")
    value = result["test/pearsonr"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("official test/pearsonr must be numeric")
    pearson = float(value)
    if not math.isfinite(pearson):
        raise RuntimeError("official test/pearsonr must be finite")
    return pearson


def _run_strict_test_phase(
    *,
    original_test: Any,
    experiment: Any,
    loaders: Mapping[str, Any],
    best_model_path: str,
    selection_path: Path,
    test_started_path: Path,
    test_completed_path: Path,
    selection_record: Mapping[str, Any],
    strict_final_test: bool,
    invocation_key: int,
    test_invocations: set[int],
    sha256_file: Any,
    prediction_exporter: Any | None = None,
) -> dict[str, Any]:
    """Seal selection and optionally consume the one permitted test pass."""

    run_dir = selection_path.parent
    legacy_metrics_path = run_dir / "epoch_test_metrics.jsonl"
    if legacy_metrics_path.exists():
        raise RuntimeError(
            "strict evaluation cannot reuse a directory with legacy test metrics: "
            f"{legacy_metrics_path}"
        )
    prior_prediction_dirs = [
        path for path in run_dir.rglob("test_predictions") if path.is_dir()
    ]
    if prior_prediction_dirs:
        raise RuntimeError(
            "strict evaluation cannot reuse a directory with prior test predictions: "
            f"{prior_prediction_dirs[0]}"
        )
    _publish_json_create_if_absent(selection_path, selection_record)
    if not strict_final_test:
        return {}
    if invocation_key in test_invocations:
        raise RuntimeError("strict experiment was already evaluated")

    checkpoint_path = Path(best_model_path)
    expected_checkpoint_sha256 = selection_record.get("checkpoint_sha256")
    if not isinstance(expected_checkpoint_sha256, str):
        raise RuntimeError("strict selection is missing checkpoint_sha256")
    test_started_payload = {
        "schema_version": "1.0",
        "evaluation_mode": "final_test",
        "selection_sha256": sha256_file(selection_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "test_evaluations": 1,
    }
    _create_test_start_marker(test_started_path, test_started_payload)
    test_invocations.add(invocation_key)

    result = original_test(experiment, loaders, best_model_path)
    if not isinstance(result, Mapping):
        raise RuntimeError("official strict test result is not a mapping")
    test_pearson = _extract_official_test_pearson([result])
    actual_checkpoint_sha256 = sha256_file(checkpoint_path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise RuntimeError("strict checkpoint hash changed during test")

    prediction_export = None
    # The facade owns the loader-to-subject-ID adapter.  Older lightweight
    # fixtures do not provide it, so preserve their marker behavior while
    # real runs export the sealed test predictions here.
    if callable(prediction_exporter):
        prediction_export = prediction_exporter(
            experiment=experiment,
            loaders=loaders,
            checkpoint_path=checkpoint_path,
            output_path=run_dir / "predictions" / "test.jsonl",
            seed=int(getattr(experiment, "seed", 0)),
        )

    _create_test_completed_marker(
        test_completed_path,
        {
            "schema_version": "1.0",
            "evaluation_mode": "final_test",
            "selection_sha256": sha256_file(selection_path),
            "checkpoint_sha256_after_test": actual_checkpoint_sha256,
            "test_pearsonr": test_pearson,
            "test_evaluations": 1,
            "prediction_export": prediction_export,
        },
    )
    return dict(result)


def _selected_validation_checkpoint_epoch(
    results: Sequence[Mapping[str, Any]],
) -> int | None:
    """Select an epoch only from explicitly recorded validation Pearson values."""

    candidates: list[tuple[float, int]] = []
    for result in results:
        records = result.get("epoch_metrics")
        if not isinstance(records, (list, tuple)):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            metric = record.get("val/pearsonr", record.get("pearsonr"))
            epoch = record.get("epoch")
            if isinstance(metric, bool) or isinstance(epoch, bool):
                continue
            if not isinstance(metric, (int, float)) or not isinstance(epoch, int):
                continue
            if math.isfinite(float(metric)):
                candidates.append((float(metric), epoch))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _head_metadata(
    reve: Any,
    *,
    head_variant: str,
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_mix_alpha: float = 0.0,
    mean_gradient_scale: float = 0.5,
    correction_gradient_scale: float = 1.0,
    correlation_loss_lambda: float = 0.0,
    robust_loss: str = "mse",
    target_scaler_mode: str = "none",
    data_mode: str = "manifest",
    manifest_path: Path | None = None,
    manifest_digest: str | None = None,
    provenance_path: Path | None = None,
    provenance_digest: str | None = None,
    timeline_count: int | None = None,
    rows: int | None = None,
    two_stage_finetune: bool = False,
    two_stage_warmup_epochs: int = 3,
    two_stage_unfreeze_last_blocks: int = 1,
    two_stage_encoder_gradient_scale: float = 0.1,
    augmentation_consistency: bool = False,
    augmentation_consistency_lambda: float = 0.0,
    augmentation_noise_scale: float = AUGMENTATION_CONSISTENCY_NOISE_SCALE,
    continued_pretraining: bool = False,
    pretraining_epochs: int = 1,
    pretraining_mask_fraction: float = 0.15,
    pretraining_mask_block_samples: int = 20,
    pretraining_learning_rate: float = 1e-5,
    pretraining_weight_decay: float = 0.05,
    pretraining_max_batches: int | None = None,
    seeds: Sequence[int],
    launch_command: str,
) -> dict[str, Any]:
    """Build stable run metadata shared by every head variant."""

    for name, value in (
        ("mean_gradient_scale", mean_gradient_scale),
        ("correction_gradient_scale", correction_gradient_scale),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 2.0:
            raise ValueError(f"{name} must be finite and in [0, 2]")
    if not math.isfinite(float(correlation_loss_lambda)) or not 0.0 <= float(correlation_loss_lambda) <= 0.1:
        raise ValueError("correlation_loss_lambda must be finite and in [0, 0.1]")
    if not math.isfinite(float(augmentation_consistency_lambda)) or not 0.0 <= float(augmentation_consistency_lambda) <= 0.1:
        raise ValueError("augmentation_consistency_lambda must be finite and in [0, 0.1]")
    if not math.isfinite(float(augmentation_noise_scale)) or not 0.0 <= float(augmentation_noise_scale) <= 0.1:
        raise ValueError("augmentation_noise_scale must be finite and in [0, 0.1]")
    if continued_pretraining:
        from ..training.continued_pretraining import ContinuedPretrainingConfig

        ContinuedPretrainingConfig(
            epochs=pretraining_epochs,
            mask_fraction=pretraining_mask_fraction,
            mask_block_samples=pretraining_mask_block_samples,
            learning_rate=pretraining_learning_rate,
            weight_decay=pretraining_weight_decay,
            max_batches=(
                None if pretraining_max_batches == 0 else pretraining_max_batches
            ),
        )

    query_initialization = {
        "mean_linear": "neuralbench_default",
        "mean_linear_copy": "not_applicable",
        "mean_linear_detached": "not_applicable",
        "mean_linear_warmup": "not_applicable",
        "mean_linear_gradient_scaled": "not_applicable",
        "mean_linear_probe_scaled": "not_applicable",
        "mean_anchor": "train_dummy_final_token_mean",
        "mean_residual": "train_dummy_final_token_mean",
        "mean_vector_anchor": "train_dummy_final_token_mean",
        "mean_mlp_residual": "not_applicable",
        "mean_stats_residual": "not_applicable",
        "mean_stats_residual_detached": "not_applicable",
        "mean_stats_residual_gradient_scaled": "not_applicable",
        "mean_stats_probe_scaled": "not_applicable",
        "mean_stats_attention_residual": "train_dummy_final_token_mean",
        "mean_attention_gated": "train_dummy_final_token_mean",
        "global_stats_residual": "not_applicable",
        "mean_rich_stats_residual": "not_applicable",
        "mean_rich_stats_gradient_routes": "not_applicable",
        "mean_anchor_ensemble": "not_applicable",
        "mean_reliability_shrinkage": "not_applicable",
        "mean_reliability_stable": "not_applicable",
        "grouped_rich_stats_shrinkage": "not_applicable",
        "grouped_stats_shared_gate": "not_applicable",
        "temporal_pyramid_stats": "not_applicable",
        "mean_covariance_residual": "not_applicable",
        "multi_query_rich_stats": "signed_basis_pm_e0",
        "mean_layer_linear": "not_applicable",
        "mean_layer_mix": "not_applicable",
        "mean_layer_mix_fixed": "not_applicable",
        "last_avg": "upstream_random_unused",
        "last_tuned": "train_dummy_final_token_mean",
        "last": "upstream_random",
        "all": "upstream_random",
    }[head_variant]
    is_default_head = head_variant == "mean_linear"
    is_local_head = head_variant in tuple(reve.LOCAL_HEAD_VARIANTS)
    if data_mode not in {"manifest", "full", "selective_task"}:
        raise ValueError(f"unsupported data mode: {data_mode!r}")
    if data_mode == "manifest" and manifest_path is not None and provenance_path is None:
        provenance_path = manifest_path
        provenance_digest = manifest_digest
    metadata: dict[str, Any] = {
        "head_variant": head_variant,
        "layer_index": int(layer_index),
        "layer_indices": None if layer_indices is None else [int(index) for index in layer_indices],
        "layer_index_convention": "positive_1_based_or_negative_final_relative",
        "head_source": (
            "neuralbench_default"
            if is_default_head
            else "local_mean_anchor"
            if head_variant == "mean_anchor"
            else "local_mean_residual"
            if head_variant == "mean_residual"
            else "local_mean_vector_anchor"
            if head_variant == "mean_vector_anchor"
            else "local_mean_mlp_residual"
            if head_variant == "mean_mlp_residual"
            else "local_mean_stats_residual"
            if head_variant == "mean_stats_residual"
            else "local_mean_stats_residual_detached"
            if head_variant == "mean_stats_residual_detached"
            else "local_mean_stats_residual_gradient_scaled"
            if head_variant == "mean_stats_residual_gradient_scaled"
            else "local_mean_linear_detached"
            if head_variant == "mean_linear_detached"
            else "local_mean_linear_warmup"
            if head_variant == "mean_linear_warmup"
            else "local_mean_linear_gradient_scaled"
            if head_variant == "mean_linear_gradient_scaled"
            else "local_mean_linear_probe_scaled"
            if head_variant == "mean_linear_probe_scaled"
            else "local_mean_stats_probe_scaled"
            if head_variant == "mean_stats_probe_scaled"
            else "local_mean_stats_attention_residual"
            if head_variant == "mean_stats_attention_residual"
            else "local_mean_attention_gated"
            if head_variant == "mean_attention_gated"
            else "local_global_stats_residual"
            if head_variant == "global_stats_residual"
            else "local_mean_rich_stats_residual"
            if head_variant == "mean_rich_stats_residual"
            else "local_mean_rich_stats_gradient_routes"
            if head_variant == "mean_rich_stats_gradient_routes"
            else "local_mean_anchor_ensemble"
            if head_variant == "mean_anchor_ensemble"
            else "local_mean_reliability_stable"
            if head_variant == "mean_reliability_stable"
            else "local_mean_reliability_shrinkage"
            if head_variant == "mean_reliability_shrinkage"
            else "local_grouped_rich_stats_shrinkage"
            if head_variant == "grouped_rich_stats_shrinkage"
            else "local_grouped_stats_shared_gate"
            if head_variant == "grouped_stats_shared_gate"
            else "local_temporal_pyramid_stats"
            if head_variant == "temporal_pyramid_stats"
            else "local_mean_covariance_residual"
            if head_variant == "mean_covariance_residual"
            else "local_mean_layer_selection"
            if head_variant in {"mean_layer_linear", "mean_layer_mix", "mean_layer_mix_fixed"}
            else "local_mean_linear_copy"
            if is_local_head
            else "upstream_reve"
        ),
        "head_dropout": 0.0,
        "correlation_loss_lambda": float(correlation_loss_lambda),
        "correlation_loss_objective": "batch_pearson" if correlation_loss_lambda else None,
        "robust_loss": robust_loss,
        "target_scaler_mode": target_scaler_mode,
        "two_stage_finetune": bool(two_stage_finetune),
        "two_stage_warmup_epochs": int(two_stage_warmup_epochs),
        "two_stage_unfreeze_last_blocks": int(two_stage_unfreeze_last_blocks),
        "two_stage_encoder_gradient_scale": float(two_stage_encoder_gradient_scale),
        "augmentation_consistency": bool(augmentation_consistency),
        "augmentation_consistency_lambda": float(augmentation_consistency_lambda),
        "augmentation_noise_scale": float(augmentation_noise_scale),
        "augmentation_consistency_batch_size": (
            AUGMENTATION_CONSISTENCY_BATCH_SIZE if augmentation_consistency else None
        ),
        "augmentation_space": "neuro_input" if augmentation_consistency else None,
        "augmentation_scope": "train_only" if augmentation_consistency else None,
        "augmentation_pairing": (
            "same_batch_example_two_views" if augmentation_consistency else None
        ),
        "continued_pretraining": bool(continued_pretraining),
        "pretraining_epochs": int(pretraining_epochs) if continued_pretraining else None,
        "pretraining_mask_fraction": (
            float(pretraining_mask_fraction) if continued_pretraining else None
        ),
        "pretraining_mask_block_samples": (
            int(pretraining_mask_block_samples) if continued_pretraining else None
        ),
        "pretraining_learning_rate": (
            float(pretraining_learning_rate) if continued_pretraining else None
        ),
        "pretraining_weight_decay": (
            float(pretraining_weight_decay) if continued_pretraining else None
        ),
        "pretraining_max_batches": (
            int(pretraining_max_batches)
            if continued_pretraining and pretraining_max_batches not in {None, 0}
            else None
        ),
        "pretraining_source_split": "train_only" if continued_pretraining else None,
        "pretraining_objective": (
            "masked_teacher_student_embedding_mse" if continued_pretraining else None
        ),
        "pretraining_age_labels_used": False if continued_pretraining else None,
        "head_query_initialization": query_initialization,
        "head_linear_initialization": (
            "neuralbench_default"
            if is_default_head
            else "torch_nn_linear_default"
            if is_local_head
            else {
                "distribution": "truncated_normal",
                "std": reve.UPSTREAM_HEAD_INIT_STD,
                "cutoff": reve.UPSTREAM_HEAD_INIT_CUTOFF,
                "bias": 0.0,
            }
        ),
        "data_mode": data_mode,
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": manifest_digest,
        "provenance_path": str(provenance_path) if provenance_path is not None else None,
        "provenance_sha256": provenance_digest,
        "timeline_count": timeline_count if timeline_count is not None else rows,
        "rows": rows,
        "device": "eeg",
        "seeds": list(seeds),
        "launch_command": launch_command,
        "protocol": reve.PROTOCOL_CONTRACT,
        "runtime": reve.runtime_metadata(),
    }
    if head_variant == "mean_linear_detached":
        metadata.update(
            {
                "head_architecture": "mean_linear_detached_encoder",
                "query_initialization": "not_applicable",
                "encoder_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "mean_linear_warmup":
        metadata.update(
            {
                "head_architecture": "mean_linear_zero_gate_residual_warmup",
                "query_initialization": "not_applicable",
                "residual_initialization": "torch_nn_linear_default",
                "gate_initialization": 0.0,
                "baseline_encoder_gradient": "detached",
                "residual_encoder_gradient": "enabled_after_gate_update",
                "normalization": "none",
            }
        )
    if head_variant == "mean_linear_gradient_scaled":
        metadata.update(
            {
                "head_architecture": "mean_linear_gradient_scaled",
                "query_initialization": "not_applicable",
                "encoder_gradient_scale": 0.1,
                "normalization": "none",
            }
        )
    if head_variant == "mean_linear_probe_scaled":
        metadata.update(
            {
                "head_architecture": "mean_linear_probe_gradient_scaled",
                "query_initialization": "not_applicable",
                "encoder_gradient_scale": 0.1,
                "probe_gradient_scale": 10.0,
                "normalization": "none",
            }
        )
    if head_variant == "mean_anchor":
        metadata.update(
            {
                "head_architecture": "mean_anchor_train_dummy_query_residual",
                "query_initialization": "train_dummy_final_token_mean",
                "gamma_initialization": 0.0,
                "normalization": "none",
            }
        )
    if head_variant == "mean_residual":
        metadata.update(
            {
                "head_architecture": "mean_residual_zero_correction_query_attention",
                "query_initialization": "train_dummy_final_token_mean",
                "correction_initialization": "zero",
                "normalization": "none",
            }
        )
    if head_variant == "mean_vector_anchor":
        metadata.update(
            {
                "head_architecture": "mean_vector_anchor_train_dummy_query_residual",
                "query_initialization": "train_dummy_final_token_mean",
                "gamma_initialization": 0.0,
                "normalization": "none",
            }
        )
    if head_variant == "mean_mlp_residual":
        metadata.update(
            {
                "head_architecture": "mean_mlp_zero_correction",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_residual":
        metadata.update(
            {
                "head_architecture": "mean_stats_zero_correction",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_and_range",
                "correction_scale": 0.5,
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_residual_detached":
        metadata.update(
            {
                "head_architecture": "mean_stats_zero_correction_detached_statistics",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_and_range",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_residual_gradient_scaled":
        metadata.update(
            {
                "head_architecture": "mean_stats_zero_correction_gradient_scaled",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_and_range",
                "correction_scale": 0.5,
                "encoder_gradient_scale": 0.5,
                "correction_backbone_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_probe_scaled":
        metadata.update(
            {
                "head_architecture": "mean_stats_zero_correction_probe_gradient_scaled",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_and_range",
                "correction_scale": 0.5,
                "encoder_gradient_scale": 1.0,
                "probe_gradient_scale": 2.0,
                "correction_backbone_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "mean_stats_attention_residual":
        metadata.update(
            {
                "head_architecture": "mean_stats_attention_zero_correction",
                "query_initialization": "train_dummy_final_token_mean",
                "correction_initialization": "zero",
                "attention_correction_scale": 0.25,
                "stats_correction_scale": 0.5,
                "correction_features": "query_attention_residual_plus_per_feature_std_and_range",
                "normalization": "none",
            }
        )
    if head_variant == "mean_attention_gated":
        metadata.update(
            {
                "head_architecture": "mean_linear_detached_attention_scalar_gate",
                "query_initialization": "train_dummy_final_token_mean",
                "correction_initialization": "small_normal",
                "correction_scale": 0.25,
                "gamma_initialization": 0.0,
                "correction_encoder_gradient": "detached",
                "normalization": "none",
            }
        )
    if head_variant == "global_stats_residual":
        metadata.update(
            {
                "head_architecture": "mean_global_stats_zero_correction",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "global_std_range_mad_and_mean_abs",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "mean_rich_stats_residual":
        metadata.update(
            {
                "head_architecture": "mean_rich_stats_zero_correction",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_range_mad_and_mean_abs",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "multi_query_rich_stats":
        metadata.update(
            {
                "head_architecture": "multi_query_rich_stats_zero_correction",
                "query_initialization": "signed_basis_pm_e0",
                "query_count": 2,
                "temperature": 1.0,
                "correction_initialization": "zero",
                "correction_features": "two_query_weighted_mean_std_range_mad_mean_abs",
                "correction_scale": 0.5,
                "query_collapse_threshold": 1e-4,
                "normalization": "none",
            }
        )
    if head_variant == "mean_layer_linear":
        metadata.update(
            {
                "head_architecture": "mean_selected_transformer_layer_linear",
                "query_initialization": "not_applicable",
                "selected_layer_index_requested": int(layer_index),
                "layer_sequence_contract": "initial_position_plus_ordered_transformer_outputs",
                "positional_input_excluded": True,
                "normalization": "none",
            }
        )
    if head_variant == "mean_layer_mix":
        metadata.update(
            {
                "head_architecture": "mean_final_layer_zero_start_early_layer_mix",
                "query_initialization": "not_applicable",
                "selected_layer_indices_requested": (
                    None if layer_indices is None else [int(index) for index in layer_indices]
                ),
                "layer_selection": (
                    "dynamic_final_four_transformer_layers"
                    if layer_indices is None
                    else "explicit_final_relative_or_1_based_transformer_layers"
                ),
                "layer_sequence_contract": "initial_position_plus_ordered_transformer_outputs",
                "positional_input_excluded": True,
                "alpha_initialization": 0.0,
                "normalization": "none",
            }
        )
    if head_variant == "mean_layer_mix_fixed":
        metadata.update(
            {
                "head_architecture": "mean_final_layer_fixed_earlier_layer_mix",
                "query_initialization": "not_applicable",
                "selected_layer_indices_requested": (
                    None if layer_indices is None else [int(index) for index in layer_indices]
                ),
                "layer_selection": (
                    "dynamic_final_four_transformer_layers"
                    if layer_indices is None
                    else "explicit_final_relative_or_1_based_transformer_layers"
                ),
                "layer_sequence_contract": "initial_position_plus_ordered_transformer_outputs",
                "positional_input_excluded": True,
                "alpha_initialization": "fixed",
                "alpha_trainable": False,
                "fixed_alpha": float(layer_mix_alpha),
                "normalization": "none",
            }
        )
    if head_variant == "mean_anchor_ensemble":
        metadata.update(
            {
                "head_architecture": "mean_linear_anchored_rich_stats_expert_ensemble",
                "query_initialization": "not_applicable",
                "expert": "mean_rich_stats_residual",
                "gate_parameterization": "centered_sigmoid",
                "gate_initialization": 0.0,
                "gate_range": [-1.0, 1.0],
                "baseline_fallback": "mean_linear_exact_at_gate_zero",
                "normalization": "none",
            }
        )
    if head_variant in {"mean_reliability_shrinkage", "mean_reliability_stable"}:
        metadata.update(
            {
                "head_architecture": (
                    "mean_linear_input_conditioned_reliability_shrinkage"
                    if head_variant == "mean_reliability_shrinkage"
                    else "mean_linear_input_conditioned_reliability_shrinkage_with_gate_stability_regularizer"
                ),
                "query_initialization": "not_applicable",
                "reliability_features": [
                    "log1p_dispersion",
                    "log1p_mean_token_norm",
                    "active_token_fraction",
                ],
                "alpha_max": 0.5,
                "gate_parameterization": "input_conditioned_sigmoid",
                "gate_initialization": -4.0,
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_range_mad_and_mean_abs",
                "correction_scale": 1.0,
                "baseline_fallback": "mean_linear_exact_at_zero_correction",
                "normalization": "none",
            }
        )
        if head_variant == "mean_reliability_stable":
            metadata.update(
                {
                    "matched_control": "mean_reliability_shrinkage",
                    "regularizer": "gate_consistency",
                    "lambda_gate": H6_GATE_STABILITY_LAMBDA,
                    "augmentation": "identity_plus_bounded_gaussian_noise",
                    "noise_scale": H6_NOISE_SCALE,
                    "noise_seed_rule": "run_seed+1009*one_based_epoch+batch_idx",
                    "validation_test_augmentation": False,
                }
            )
    if head_variant == "grouped_rich_stats_shrinkage":
        metadata.update(
            {
                "head_architecture": "mean_grouped_rich_stats_zero_gate_shrinkage",
                "query_initialization": "not_applicable",
                "statistic_groups": ["std", "range", "mad", "mean_abs"],
                "gate_parameterization": "direct_scalar",
                "gate_initialization": 0.0,
                "projection_initialization": (
                    "linspace_-1_1_roll_group_plus_row_alternating_sign_l2_normalized_zero_bias"
                ),
                "projection_shape": "D_to_D",
                "parameter_count_formula": "D*n_outputs+n_outputs+4*(D*D+D)+4",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "grouped_stats_shared_gate":
        metadata.update(
            {
                "head_architecture": "mean_grouped_stats_shared_gate",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "statistic_groups": ["std", "range", "mad", "mean_abs"],
                "gate_parameterization": "shared_scalar",
                "gate_initialization": 0.0,
                "projection_initialization": (
                    "linspace_-1_1_roll_group_plus_row_alternating_sign_l2_normalized_zero_bias"
                ),
                "projection_shape": "D_to_D",
                "parameter_count_formula": "D*n_outputs+n_outputs+4*(D*D+D)+1",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "temporal_pyramid_stats":
        metadata.update(
            {
                "head_architecture": "mean_temporal_pyramid_stats_low_rank_residual",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero_via_up_factor",
                "segments": 2,
                "statistics": ["std", "range", "mad", "mean_abs"],
                "correction_rank": 8,
                "low_rank_parameterization": "down_then_up",
                "parameter_count_formula": "D*n_outputs+n_outputs+(8*D)*8+D*8",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "token_order_contract": "contiguous_ordered_segments",
                "normalization": "none",
            }
        )
    if head_variant == "mean_covariance_residual":
        metadata.update(
            {
                "head_architecture": "mean_diagonal_covariance_low_rank_residual",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero_via_up_factor",
                "covariance_mode": "diagonal",
                "covariance_features": "diagonal_sample_variance",
                "projection_rank": 4,
                "low_rank_parameterization": "down_then_up",
                "parameter_count_formula": "D*n_outputs+n_outputs+2*D*4",
                "correction_scale": 0.5,
                "correction_backbone_gradient": "enabled",
                "normalization": "none",
            }
        )
    if head_variant == "mean_rich_stats_gradient_routes":
        metadata.update(
            {
                "head_architecture": "mean_rich_stats_gradient_routes",
                "query_initialization": "not_applicable",
                "correction_initialization": "zero",
                "correction_features": "per_feature_std_range_mad_mean_abs",
                "correction_scale": 0.5,
                "gradient_target": "encoder_routes_only",
                "mean_gradient_scale": float(mean_gradient_scale),
                "correction_gradient_scale": float(correction_gradient_scale),
                "normalization": "none",
            }
        )
    if head_variant == "last_tuned":
        metadata.update(
            {
                "head_source": reve.LAST_TUNED_HEAD_SOURCE,
                "head_architecture": reve.LAST_TUNED_HEAD_ARCHITECTURE,
                "protocol_class": reve.LAST_TUNED_PROTOCOL_CLASS,
                "residual_initial_alpha": reve.LAST_TUNED_INITIAL_ALPHA,
                "query_initialization": query_initialization,
                "base_learning_rate": reve.LAST_TUNED_BASE_LR,
                "query_learning_rate": reve.LAST_TUNED_QUERY_LR,
                "optimizer": "AdamW",
                "weight_decay": reve.LAST_TUNED_WEIGHT_DECAY,
                "scheduler": "OneCycleLR",
                "scheduler_max_lr": list(reve.LAST_TUNED_SCHEDULER_MAX_LR),
                "scheduler_pct_start": reve.LAST_TUNED_SCHEDULER_PCT_START,
                "scheduler_anneal_strategy": "cos",
                "scheduler_div_factor": reve.LAST_TUNED_SCHEDULER_DIV_FACTOR,
                "scheduler_final_div_factor": reve.LAST_TUNED_SCHEDULER_FINAL_DIV_FACTOR,
                "scheduler_interval": "step",
                "scheduler_frequency": 1,
                "monitor": "val/pearsonr",
                "checkpoint_selection_monitor": "val/pearsonr",
                "test_pearsonr_role": "diagnostic_only",
            }
        )
    if not is_default_head and not is_local_head:
        metadata["head_source_lock"] = reve.source_lock_metadata()
    complexity_builder = getattr(reve, "head_complexity_metadata", None)
    if callable(complexity_builder):
        metadata["head_complexity"] = complexity_builder(
            head_variant,
            embed_dim=int(getattr(reve, "UPSTREAM_HEAD_HIDDEN_SIZE", 512)),
            n_outputs=1,
            layer_count=None,
        )
    else:
        metadata["complexity_status"] = "unavailable"
    return metadata


def _append_evaluation_callback(
    trainer: Any,
    *,
    evaluation_protocol: str,
    epoch_metrics_path: Path,
    seed: int | None,
    loaders: Mapping[str, Any],
    hooks: Any,
    head_variant: str = "mean_linear",
    mean_gradient_scale: float = 0.5,
    correction_gradient_scale: float = 1.0,
    swa_window: int = 0,
    target_scaler_mode: str = "none",
    continued_pretraining: bool = False,
    continued_pretraining_config: Any | None = None,
    two_stage_finetune: bool = False,
    two_stage_config: Any | None = None,
) -> None:
    """Attach exactly one training-time metric callback for the protocol."""

    if continued_pretraining:
        if continued_pretraining_config is None:
            raise RuntimeError("continued pretraining is enabled without its configuration")
        if "train" not in loaders:
            raise RuntimeError("continued pretraining requires the official train loader")
        from ..training.continued_pretraining import ContinuedPretrainingCallback

        trainer.callbacks.append(
            ContinuedPretrainingCallback(
                continued_pretraining_config,
                train_loader=loaders["train"],
                metadata_path=epoch_metrics_path.parent / "continued_pretraining.json",
                metrics_path=epoch_metrics_path.parent / "continued_pretraining_metrics.jsonl",
                run_seed=int(seed if seed is not None else 0),
            )
        )

    if two_stage_finetune:
        if two_stage_config is None:
            raise RuntimeError("two-stage fine-tuning is enabled without its configuration")
        from ..training.two_stage import TwoStageFineTuneCallback

        trainer.callbacks.append(
            TwoStageFineTuneCallback(
                two_stage_config,
                metadata_path=epoch_metrics_path.parent / "two_stage_finetuning.json",
            )
        )

    if "train" in loaders:
        trainer.callbacks.append(
            hooks.TrainAgeReferenceExporter(
                epoch_metrics_path.parent / "analysis" / "train_age_reference.jsonl",
                seed=seed,
            )
        )
        trainer.callbacks.append(
            hooks.OptimizerEvidenceExporter(
                epoch_metrics_path.parent / "optimizer.json",
            )
        )
        trainer.callbacks.append(
            hooks.ThroughputEvidenceExporter(
                epoch_metrics_path.parent / "throughput.json",
            )
        )

    if evaluation_protocol == "strict":
        trainer.callbacks.append(
            hooks.EpochValidationMetrics(
                epoch_metrics_path,
                seed=seed,
                validation_loader=loaders.get("val"),
                inverse_transform_targets=target_scaler_mode == "zscore",
            )
        )
        if head_variant in {"mean_reliability_shrinkage", "mean_reliability_stable"}:
            trainer.callbacks.append(
                hooks.ReliabilityGateAudit(
                    epoch_metrics_path.parent / "gate_validation_audit.jsonl",
                    seed=seed,
                    alpha_max=0.5,
                )
            )
        if head_variant == "mean_reliability_stable":
            trainer.callbacks.append(
                hooks.H6GateConsistencyMetrics(
                    epoch_metrics_path.parent / "gate_consistency.json",
                    seed=seed,
                    lambda_gate=H6_GATE_STABILITY_LAMBDA,
                    noise_scale=H6_NOISE_SCALE,
                )
                )
        if head_variant == "mean_rich_stats_gradient_routes":
            trainer.callbacks.append(
                hooks.H7GradientRouteAudit(
                    epoch_metrics_path.parent / "gradient_norms.json",
                    seed=seed,
                    mean_gradient_scale=mean_gradient_scale,
                    correction_gradient_scale=correction_gradient_scale,
                )
            )
        if swa_window:
            if head_variant != "mean_rich_stats_residual":
                raise ValueError("SWA screen requires the accepted mean_rich_stats_residual head")
            trainer.callbacks.append(
                hooks.SWAValidationCheckpoint(
                    epoch_metrics_path,
                    loaders["val"],
                    seed=seed,
                    window_size=swa_window,
                )
            )
        validation_loader = loaders.get("val")
        if validation_loader is not None:
            trainer.callbacks.append(
                hooks.ValidationPredictionExporter(
                    epoch_metrics_path.parent / "predictions" / "validation.jsonl",
                    validation_loader,
                    seed=seed,
                )
            )
        return
    if evaluation_protocol != "legacy":
        raise ValueError(f"unsupported evaluation protocol: {evaluation_protocol!r}")
    if "test" not in loaders:
        raise RuntimeError("official test loader was not captured")
    trainer.callbacks.append(
        hooks.EpochTestPearson(loaders["test"], epoch_metrics_path, seed=seed)
    )


# ---------------------------------------------------------------------------
# Official run patch lifecycle
# ---------------------------------------------------------------------------


def _patch_official_components(
    manifest_path: Path | None,
    data_root: Path,
    epoch_metrics_path: Path,
    selection_path: Path,
    *,
    head_variant: str = "mean_linear",
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_mix_alpha: float = 0.0,
    head_dropout: float = 0.0,
    mean_gradient_scale: float = 0.5,
    correction_gradient_scale: float = 1.0,
    swa_window: int = 0,
    correlation_loss_lambda: float = 0.0,
    robust_loss: str = "mse",
    target_scaler_mode: str = "none",
    seeds: Sequence[int] = (33,),
    evaluation_protocol: str = "strict",
    strict_final_test: bool = False,
    two_stage_finetune: bool = False,
    two_stage_warmup_epochs: int = 3,
    two_stage_unfreeze_last_blocks: int = 1,
    two_stage_encoder_gradient_scale: float = 0.1,
    augmentation_consistency: bool = False,
    augmentation_consistency_lambda: float = 0.0,
    augmentation_noise_scale: float = AUGMENTATION_CONSISTENCY_NOISE_SCALE,
    continued_pretraining: bool = False,
    pretraining_epochs: int = 1,
    pretraining_mask_fraction: float = 0.15,
    pretraining_mask_block_samples: int = 20,
    pretraining_learning_rate: float = 1e-5,
    pretraining_weight_decay: float = 0.05,
    pretraining_max_batches: int = 0,
    data_mode: str = "manifest",
    provenance_path: Path | None = None,
    acquisition_provenance_path: Path | None = None,
    acquisition_provenance_sha256: str | None = None,
    timeline_count: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    final_results: list[dict[str, Any]] | None = None,
    evidence_recorder: Any | None = None,
    hooks: Any,
) -> dict[str, Any]:
    """Install the fixed-manifest and optional upstream-head patches.

    The patch is deliberately scoped to one ``run_benchmark`` call.  The
    official package remains untouched after :func:`_restore_official_components`
    runs, which is important when several variants are evaluated in one Python
    process.
    """

    evaluation_protocol, strict_final_test = hooks.validate_evaluation_options(
        evaluation_protocol,
        strict_final_test=strict_final_test,
    )
    reve = hooks._load_reve_helpers()

    continued_pretraining_config = None
    if continued_pretraining:
        if evaluation_protocol != "strict":
            raise ValueError("continued pretraining requires strict evaluation")
        if head_variant != "mean_linear":
            raise ValueError("continued pretraining screen currently requires mean_linear")
        if augmentation_consistency or two_stage_finetune:
            raise ValueError(
                "continued pretraining must be screened as a standalone training factor"
            )
        from ..training.continued_pretraining import ContinuedPretrainingConfig

        if isinstance(pretraining_max_batches, bool) or not isinstance(pretraining_max_batches, int):
            raise ValueError("pretraining_max_batches must be an integer")
        if pretraining_max_batches < 0:
            raise ValueError("pretraining_max_batches must be non-negative")
        continued_pretraining_config = ContinuedPretrainingConfig(
            epochs=pretraining_epochs,
            mask_fraction=pretraining_mask_fraction,
            mask_block_samples=pretraining_mask_block_samples,
            learning_rate=pretraining_learning_rate,
            weight_decay=pretraining_weight_decay,
            max_batches=(None if pretraining_max_batches == 0 else pretraining_max_batches),
        )

    two_stage_config = None
    if two_stage_finetune:
        from ..training.two_stage import (
            TwoStageFineTuneConfig,
            validate_two_stage_options,
        )

        validate_two_stage_options(
            head_variant=head_variant,
            data_mode=data_mode,
            evaluation_protocol=evaluation_protocol,
        )
        two_stage_config = TwoStageFineTuneConfig(
            warmup_epochs=two_stage_warmup_epochs,
            unfreeze_last_blocks=two_stage_unfreeze_last_blocks,
            encoder_gradient_scale=two_stage_encoder_gradient_scale,
        )

    reve.validate_head_variant(head_variant)
    if head_dropout != 0.0:
        raise ValueError(f"the upstream REVE comparison fixes head dropout at 0.0; got {head_dropout}")
    for name, value in (
        ("mean_gradient_scale", mean_gradient_scale),
        ("correction_gradient_scale", correction_gradient_scale),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 2.0:
            raise ValueError(f"{name} must be finite and in [0, 2]")
    if isinstance(swa_window, bool) or not isinstance(swa_window, int) or swa_window not in {0, 3, 5}:
        raise ValueError("swa_window must be one of 0, 3, or 5")
    if swa_window and (head_variant != "mean_rich_stats_residual" or evaluation_protocol != "strict"):
        raise ValueError("SWA requires strict evaluation of mean_rich_stats_residual")
    if not math.isfinite(float(correlation_loss_lambda)) or not 0.0 <= float(correlation_loss_lambda) <= 0.1:
        raise ValueError("correlation_loss_lambda must be finite and in [0, 0.1]")
    if not math.isfinite(float(augmentation_consistency_lambda)) or not 0.0 <= float(augmentation_consistency_lambda) <= 0.1:
        raise ValueError("augmentation_consistency_lambda must be finite and in [0, 0.1]")
    if not math.isfinite(float(augmentation_noise_scale)) or not 0.0 <= float(augmentation_noise_scale) <= 0.1:
        raise ValueError("augmentation_noise_scale must be finite and in [0, 0.1]")
    if augmentation_consistency and augmentation_consistency_lambda == 0.0:
        raise ValueError("augmentation consistency requires a positive lambda")
    if correlation_loss_lambda and (
        head_variant != "mean_rich_stats_residual" or evaluation_protocol != "strict"
    ):
        raise ValueError("correlation loss requires strict evaluation of mean_rich_stats_residual")
    if robust_loss not in {"mse", "smooth_l1"}:
        raise ValueError("robust_loss must be 'mse' or 'smooth_l1'")
    if robust_loss != "mse" and (
        head_variant != "mean_rich_stats_residual" or evaluation_protocol != "strict"
    ):
        raise ValueError("smooth_l1 requires strict evaluation of mean_rich_stats_residual")
    if target_scaler_mode not in {"none", "zscore"}:
        raise ValueError("target_scaler_mode must be 'none' or 'zscore'")
    if target_scaler_mode == "zscore" and (
        head_variant != "mean_rich_stats_residual" or evaluation_protocol != "strict"
    ):
        raise ValueError("target z-score requires strict evaluation of mean_rich_stats_residual")
    resolved_seeds = hooks.validate_seeds(seeds)
    if head_variant == "mean_reliability_stable" and evaluation_protocol != "strict":
        raise ValueError("mean_reliability_stable requires strict evaluation")
    if data_mode not in {"manifest", "full", "selective_task"}:
        raise ValueError(f"unsupported data mode: {data_mode!r}")
    if data_mode == "manifest" and manifest_path is None:
        raise ValueError("manifest data mode requires manifest_path")
    if data_mode in {"full", "selective_task"} and manifest_path is not None:
        raise ValueError("non-manifest data mode must not receive manifest_path")
    if data_mode in {"full", "selective_task"} and provenance_path is None:
        raise ValueError("non-manifest data mode requires provenance_path")
    if data_mode == "selective_task" and (
        acquisition_provenance_path is None or acquisition_provenance_sha256 is None
    ):
        raise ValueError("selective task mode requires acquisition provenance")
    if data_mode in {"full", "selective_task"}:
        assert provenance_path is not None
        provenance_path.unlink(missing_ok=True)

    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    neuralset_study = None
    original_study_all_timelines = None
    selective_eeglab_reader: dict[str, Any] | None = None
    if data_mode == "selective_task":
        from neuralset.events import study as neuralset_study

        original_study_all_timelines = neuralset_study.Study._all_timelines

    timelines: list[dict[str, Any]] | None = None
    if data_mode == "manifest":
        assert manifest_path is not None
        timelines = hooks.load_manifest_timelines(manifest_path, data_root)

    original_iter_timelines = shirazi2024hbn.Shirazi2024Hbn.iter_timelines
    original_info = shirazi2024hbn.Shirazi2024Hbn._info
    original_prepare = Data.prepare
    original_test = Experiment._test
    original_setup_run = Experiment.setup_run
    original_prepare_pl_module = Experiment.prepare_pl_module
    original_setup_trainer = Experiment.setup_trainer

    def iter_manifest_timelines(_study: Any) -> Iterable[dict[str, Any]]:
        assert timelines is not None
        return iter(timelines)

    def selective_all_timelines(study_instance: Any) -> list[dict[str, Any]]:
        """Allow the official study iterator to describe a task-only tree.

        NeuralFetch's static ``_info.num_timelines`` describes all HBN tasks,
        while selective acquisition intentionally contains only RestingState.
        Keep the iterator and ``_info`` unchanged outside this temporary call,
        but bypass that full-dataset cardinality assertion for the selective
        study so the actual discovered timelines can be audited downstream.
        """

        assert original_study_all_timelines is not None
        if not isinstance(study_instance, shirazi2024hbn.Shirazi2024Hbn):
            return original_study_all_timelines(study_instance)
        study_class = study_instance.__class__
        original_info = study_class._info
        try:
            study_class._info = None
            return original_study_all_timelines(study_instance)
        finally:
            study_class._info = original_info

    captured_loaders: dict[int, dict[str, Any]] = {}
    provenance_state_by_data: dict[int, dict[str, Any]] = {}
    patched_brain_modules: list[dict[str, Any]] = []
    patched_h6_training_steps: list[dict[str, Any]] = []
    patched_augmentation_consistency_training_steps: list[dict[str, Any]] = []
    tuning_metadata_by_experiment: dict[int, dict[str, Any]] = {}
    test_invocations: set[int] = set()

    def persist_run_metadata(
        experiment: Any,
        metadata: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> None:
        """Persist metadata to every available official/canonical run path."""

        for path in _run_metadata_paths(experiment, epoch_metrics_path.parent):
            payload: dict[str, Any] = {}
            if not replace and path.is_file():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(loaded, Mapping):
                    raise ValueError("run_metadata.json must contain a JSON object")
                payload.update(loaded)
            payload.update(metadata)
            _replace_json_atomic(path, payload)

    def prepare_and_capture(data: Any) -> dict[str, Any]:
        source_study = None
        if data_mode in {"full", "selective_task"}:
            study = getattr(data, "study", None)
            source_study = _find_nested_study(
                study,
                shirazi2024hbn.Shirazi2024Hbn,
            )
            if source_study is None and isinstance(
                study,
                shirazi2024hbn.Shirazi2024Hbn,
            ):
                source_study = study
        loaders = original_prepare(data)
        captured_loaders[id(data)] = loaders
        if data_mode in {"full", "selective_task"}:
            if source_study is not None:
                actual_timelines = getattr(source_study, "_timelines", None)
                if actual_timelines is None:
                    all_timelines = getattr(source_study, "_all_timelines", None)
                    actual_timelines = all_timelines() if callable(all_timelines) else None
            else:
                study = getattr(data, "study", None)
                actual_timelines = getattr(study, "_timelines", None)
            if actual_timelines is None:
                raise RuntimeError(
                    "full-data Data.prepare did not expose the source study timelines"
                )
            normalized = hooks._canonical_full_data_timelines(
                actual_timelines,
                expected_task="task-RestingState" if data_mode == "selective_task" else None,
            )
            payload, raw = hooks._full_data_provenance_payload(
                data_root=data_root,
                timelines=normalized,
                data_mode=data_mode,
            )
            assert provenance_path is not None
            try:
                _replace_bytes_atomic(
                    provenance_path,
                    raw,
                    remove_final_on_failure=True,
                )
                final_bytes = provenance_path.read_bytes()
                provenance_digest = hooks._sha256_bytes(final_bytes)
            except BaseException:
                provenance_path.unlink(missing_ok=True)
                raise
            state = {
                "data_mode": data_mode,
                "timeline_count": payload["timeline_count"],
                "provenance_path": str(provenance_path.resolve()),
                "provenance_sha256": provenance_digest,
            }
            if data_mode == "selective_task":
                assert acquisition_provenance_path is not None
                assert acquisition_provenance_sha256 is not None
                state.update(
                    {
                        "acquisition_provenance_path": str(
                            acquisition_provenance_path.resolve()
                        ),
                        "acquisition_provenance_sha256": acquisition_provenance_sha256,
                    }
                )
            provenance_state_by_data[id(data)] = state
        return loaders

    def test_and_capture(
        self: Any,
        loaders: dict[str, Any],
        best_model_path: str | None,
    ) -> dict[str, Any]:
        if evaluation_protocol == "legacy":
            if evidence_recorder is None:
                result = original_test(self, loaders, best_model_path)
            else:
                with evidence_recorder.phase("test_diagnostic"):
                    result = original_test(self, loaders, best_model_path)
            if final_results is not None:
                final_results.append(
                    hooks._capture_test_result(
                        result,
                        head_variant=head_variant,
                        experiment_id=id(self),
                        tuning_metadata_by_experiment=tuning_metadata_by_experiment,
                    )
                )
            return result

        uid_folder = self.infra.uid_folder()
        if uid_folder is None:
            raise RuntimeError("strict evaluation requires an official uid folder")
        if best_model_path is None:
            raise RuntimeError("strict evaluation requires a selected checkpoint")
        source_state = provenance_state_by_data.get(id(self.data))
        if data_mode in {"full", "selective_task"} and source_state is None:
            raise RuntimeError("non-manifest strict evaluation has no post-prepare provenance")
        selection_manifest_path = manifest_path if data_mode == "manifest" else None
        selection_provenance_path = (
            Path(source_state["provenance_path"])
            if source_state is not None
            else manifest_path
        )
        selection_timeline_count = (
            int(source_state["timeline_count"])
            if source_state is not None
            else len(timelines or [])
        )
        selection_record = hooks._build_strict_selection_record(
            checkpoint_path=Path(best_model_path),
            official_config_path=uid_folder / "config.yaml",
            manifest_path=selection_manifest_path,
            provenance_path=selection_provenance_path,
            acquisition_provenance_path=(
                Path(source_state["acquisition_provenance_path"])
                if data_mode == "selective_task" and source_state is not None
                else None
            ),
            acquisition_provenance_sha256=(
                str(source_state["acquisition_provenance_sha256"])
                if data_mode == "selective_task" and source_state is not None
                else None
            ),
            data_mode=data_mode,
            timeline_count=selection_timeline_count,
            validation_history_path=epoch_metrics_path,
            selection_monitor="val/pearsonr",
            selection_mode="max",
            seed=int(getattr(self, "seed", 0)),
            head_variant=head_variant,
            strict_final_test=strict_final_test,
            sha256_file=hooks._sha256_file,
        )
        def _export_final_predictions(
            *,
            experiment: Any,
            loaders: Mapping[str, Any],
            checkpoint_path: Path,
            output_path: Path,
            seed: int,
        ) -> dict[str, Any] | None:
            brain_module = getattr(experiment, "_brain_module", None)
            trainer = getattr(experiment, "trainer", None)
            if trainer is None:
                trainer = getattr(experiment, "_trainer", None)
            if trainer is None and brain_module is not None:
                trainer = getattr(brain_module, "trainer", None)
            if trainer is None or brain_module is None:
                return None
            return hooks._export_prediction_file(
                trainer=trainer,
                pl_module=brain_module,
                loader=loaders["test"],
                output_path=output_path,
                split="test",
                seed=seed,
                selected_checkpoint_path=checkpoint_path,
            )

        def _strict_test() -> Any:
            return hooks._run_strict_test_phase(
                original_test=original_test,
                experiment=self,
                loaders=loaders,
                best_model_path=best_model_path,
                selection_path=selection_path,
                test_started_path=selection_path.parent / "test_started.json",
                test_completed_path=selection_path.parent / "test_completed.json",
                selection_record=selection_record,
                strict_final_test=strict_final_test,
                invocation_key=id(self),
                test_invocations=test_invocations,
                sha256_file=hooks._sha256_file,
                prediction_exporter=_export_final_predictions,
            )
        if evidence_recorder is None:
            result = _strict_test()
        else:
            with evidence_recorder.phase(
                "test_final" if strict_final_test else "validation_selection"
            ):
                result = _strict_test()
        if strict_final_test and final_results is not None:
            final_results.append(
                hooks._capture_test_result(
                    result,
                    head_variant=head_variant,
                    experiment_id=id(self),
                    tuning_metadata_by_experiment=tuning_metadata_by_experiment,
                )
            )
        return result

    def setup_with_evaluation_callbacks(self: Any, is_test: bool = False) -> Any:
        trainer = original_setup_trainer(self, is_test=is_test)
        if not is_test:
            loaders = captured_loaders.get(id(self.data))
            if loaders is None:
                raise RuntimeError("official loaders were not captured")
            _append_evaluation_callback(
                trainer,
                evaluation_protocol=evaluation_protocol,
                epoch_metrics_path=epoch_metrics_path,
                seed=getattr(self, "seed", None),
                loaders=loaders,
                head_variant=head_variant,
                mean_gradient_scale=mean_gradient_scale,
                correction_gradient_scale=correction_gradient_scale,
                swa_window=swa_window,
                target_scaler_mode=target_scaler_mode,
                continued_pretraining=continued_pretraining,
                continued_pretraining_config=continued_pretraining_config,
                two_stage_finetune=two_stage_finetune,
                two_stage_config=two_stage_config,
                hooks=hooks,
            )
        return trainer

    def setup_with_metadata(self: Any) -> Any:
        # The standard REVE YAML wrapper is mean-pooling plus a linear probe.
        # For upstream variants, replace only that downstream config; the
        # already-built official NtReve encoder still comes from NeuralBench.
        if head_variant != "mean_linear":
            hooks._set_frozen_experiment_field(
                self,
                "downstream_model_wrapper",
                reve.make_upstream_reve_wrapper(
                    variant=head_variant,
                    dropout=head_dropout,
                    mean_gradient_scale=mean_gradient_scale,
                    correction_gradient_scale=correction_gradient_scale,
                    layer_index=layer_index,
                    layer_indices=layer_indices,
                    layer_mix_alpha=layer_mix_alpha,
                ),
            )
        if head_variant == "last_tuned":
            # NeuralBench expresses the actual checkpoint criterion through
            # ``trainer_config``.  Record its resolved tuning counterpart on
            # this run instance so the separate tuning validator can reject a
            # relabeling of the diagnostic test callback as a selector.
            hooks._set_frozen_experiment_field(
                self,
                "checkpoint_selection",
                {
                    "monitor": "val/pearsonr",
                    "mode": "max",
                    "test_pearsonr_role": "diagnostic_only",
                },
            )
        hooks._set_frozen_experiment_field(self, "save_test_predictions", True)
        # Keep the selected checkpoint and raw prediction cache available for
        # post-run hashing/export. This does not affect training or selection.
        hooks._set_frozen_experiment_field(self, "delete_checkpoints_on_exit", False)
        if target_scaler_mode == "zscore":
            hooks._set_frozen_experiment_field(self, "target_scaler", TrainingOnlyTargetZScore())
        result = original_setup_run(self)
        payload = dict(run_metadata or {})
        payload.update(
            {
                    "head_variant": head_variant,
                    "head_dropout": float(head_dropout),
                    "seed": int(self.seed),
                    "data_seed": hooks._get_attr_or_key(self.data, "seed"),
                    "protocol": reve.PROTOCOL_CONTRACT,
                    "evaluation_protocol": evaluation_protocol,
                    "strict_final_test": bool(strict_final_test),
                    "correlation_loss_lambda": float(correlation_loss_lambda),
                    "correlation_loss_objective": (
                        "batch_pearson" if correlation_loss_lambda else None
                    ),
                    "robust_loss": robust_loss,
                    "target_scaler_mode": target_scaler_mode,
                    "two_stage_finetune": bool(two_stage_finetune),
                    "two_stage_warmup_epochs": (
                        int(two_stage_warmup_epochs) if two_stage_finetune else None
                    ),
                    "two_stage_unfreeze_last_blocks": (
                        int(two_stage_unfreeze_last_blocks) if two_stage_finetune else None
                    ),
                    "two_stage_encoder_gradient_scale": (
                        float(two_stage_encoder_gradient_scale) if two_stage_finetune else None
                    ),
                    "augmentation_consistency": bool(augmentation_consistency),
                    "augmentation_consistency_lambda": (
                        float(augmentation_consistency_lambda)
                        if augmentation_consistency
                        else 0.0
                    ),
                    "augmentation_noise_scale": (
                        float(augmentation_noise_scale)
                        if augmentation_consistency
                        else None
                    ),
                    "augmentation_consistency_batch_size": (
                        AUGMENTATION_CONSISTENCY_BATCH_SIZE
                        if augmentation_consistency
                        else None
                    ),
                    "augmentation_space": (
                        "neuro_input" if augmentation_consistency else None
                    ),
                    "augmentation_scope": "train_only" if augmentation_consistency else None,
                    "continued_pretraining": bool(continued_pretraining),
                    "pretraining_epochs": (
                        int(pretraining_epochs) if continued_pretraining else None
                    ),
                    "pretraining_mask_fraction": (
                        float(pretraining_mask_fraction) if continued_pretraining else None
                    ),
                    "pretraining_mask_block_samples": (
                        int(pretraining_mask_block_samples)
                        if continued_pretraining
                        else None
                    ),
                    "pretraining_learning_rate": (
                        float(pretraining_learning_rate)
                        if continued_pretraining
                        else None
                    ),
                    "pretraining_weight_decay": (
                        float(pretraining_weight_decay)
                        if continued_pretraining
                        else None
                    ),
                    "pretraining_max_batches": (
                        int(pretraining_max_batches)
                        if continued_pretraining and pretraining_max_batches
                        else None
                    ),
                    "pretraining_source_split": (
                        "train_only" if continued_pretraining else None
                    ),
                    "pretraining_objective": (
                        "masked_teacher_student_embedding_mse"
                        if continued_pretraining
                        else None
                    ),
                    "pretraining_age_labels_used": (
                        False if continued_pretraining else None
                    ),
                    "test_access_policy": (
                        "single_use_predeclared"
                        if strict_final_test
                        else "withheld"
                        if evaluation_protocol == "strict"
                        else "epoch_diagnostic"
                    ),
            }
        )
        persist_run_metadata(self, payload, replace=True)
        return result

    def persist_tuning_metadata(self: Any, metadata: Mapping[str, Any]) -> None:
        """Merge late-bound query and optimizer details into run metadata."""

        persist_run_metadata(self, metadata)

    def persist_provenance_metadata(self: Any) -> None:
        if data_mode not in {"full", "selective_task"}:
            return
        state = provenance_state_by_data.get(id(self.data))
        if state is None:
            raise RuntimeError("non-manifest experiment has no post-prepare provenance")
        persist_run_metadata(self, state)

    def prepare_with_protocol(
        self: Any,
        train_loader: Any,
        val_loader: Any = None,
    ) -> Any:
        result = original_prepare_pl_module(self, train_loader, val_loader)
        persist_provenance_metadata(self)
        brain_module = getattr(self, "_brain_module", None)
        model = getattr(brain_module, "model", brain_module)
        if model is not None and callable(getattr(model, "parameters", None)):
            head = getattr(model, "head", None)
            try:
                parameter_buckets = parameter_buckets_from_model(model, head=head)
                declared_head = (
                    run_metadata.get("head_complexity")
                    if isinstance(run_metadata, Mapping)
                    else None
                )
                declared_head_count = (
                    declared_head.get("parameter_count")
                    if isinstance(declared_head, Mapping)
                    else None
                )
                if (
                    isinstance(declared_head_count, int)
                    and not isinstance(declared_head_count, bool)
                ):
                    parameter_buckets = add_declared_head_bucket(
                        parameter_buckets,
                        parameter_count=declared_head_count,
                    )
            except (TypeError, ValueError, RuntimeError) as accounting_error:
                persist_tuning_metadata(
                    self,
                    {
                        "parameter_accounting_status": "unavailable",
                        "parameter_accounting_error": str(accounting_error),
                    },
                )
            else:
                persist_tuning_metadata(
                    self,
                    {
                        "parameter_buckets": parameter_buckets,
                        "parameter_accounting_status": "complete",
                    },
                )
        if head_variant in {"grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable"}:
            brain_module = getattr(self, "_brain_module", None)
            grouped_head = None
            expected_class_name = {
                "grouped_rich_stats_shrinkage": "GroupedRichStatsShrinkageHead",
                "grouped_stats_shared_gate": "GroupedStatsSharedGateHead",
                "temporal_pyramid_stats": "TemporalPyramidStatsResidualHead",
                "mean_covariance_residual": "MeanCovarianceResidualHead",
                "mean_anchor_ensemble": "MeanAnchorEnsembleHead",
                "mean_reliability_shrinkage": "MeanReliabilityShrinkageHead",
                "mean_reliability_stable": "MeanReliabilityShrinkageHead",
            }[head_variant]
            if brain_module is not None and hasattr(brain_module, "modules"):
                grouped_head = next(
                    (
                        module
                        for module in brain_module.modules()
                        if module.__class__.__name__ == expected_class_name
                    ),
                    None,
                )
            if grouped_head is None or not callable(getattr(grouped_head, "metadata", None)):
                raise RuntimeError(f"{head_variant} model did not expose its head metadata")
            head_metadata = dict(grouped_head.metadata())
            head_metadata["parameter_count"] = sum(
                parameter.numel() for parameter in grouped_head.parameters()
            )
            persist_tuning_metadata(
                self,
                {
                    "head_metadata": head_metadata,
                    "head_parameter_count": head_metadata["parameter_count"],
                },
            )
            if head_variant == "mean_reliability_stable":
                _patch_h6_training_step(
                    brain_module,
                    run_seed=int(getattr(self, "seed", 0)),
                    patched_modules=patched_h6_training_steps,
                )
        if head_variant in {"mean_layer_linear", "mean_layer_mix", "mean_layer_mix_fixed"}:
            brain_module = getattr(self, "_brain_module", None)
            expected_class_name = (
                "MeanLayerLinearHead"
                if head_variant == "mean_layer_linear"
                else "MeanLayerMixHead"
                if head_variant == "mean_layer_mix"
                else "MeanLayerMixFixedHead"
            )
            layer_head = None
            if brain_module is not None and hasattr(brain_module, "modules"):
                layer_head = next(
                    (
                        module
                        for module in brain_module.modules()
                        if module.__class__.__name__ == expected_class_name
                    ),
                    None,
                )
            if layer_head is None or not callable(getattr(layer_head, "metadata", None)):
                raise RuntimeError(f"{head_variant} model did not expose its head metadata")
            head_metadata = dict(layer_head.metadata())
            head_metadata["parameter_count"] = sum(
                parameter.numel() for parameter in layer_head.parameters()
            )
            persist_tuning_metadata(
                self,
                {
                    "head_metadata": head_metadata,
                    "head_parameter_count": head_metadata["parameter_count"],
                },
            )
        if head_variant == "last_tuned":
            brain_module = getattr(self, "_brain_module", None)
            model = getattr(brain_module, "model", None)
            tuning_model = reve._resolve_last_tuned_model(model)
            query_metadata = getattr(getattr(tuning_model, "head", None), "tuning_metadata", None)
            if not isinstance(query_metadata, Mapping):
                raise RuntimeError("last_tuned prepared model did not expose tuning metadata")
            optimizer_metadata = reve.last_tuned_optimizer_metadata(tuning_model)
            _patch_last_tuned_configure_optimizers(brain_module, patched_brain_modules, hooks=hooks)
            reve.validate_last_tuned_protocol(head_variant, experiment=self, optimizer_config=optimizer_metadata)
            tuning_metadata = hooks._last_tuned_report_metadata(query_metadata=query_metadata, optimizer_config=optimizer_metadata)
            tuning_metadata_by_experiment[id(self)] = tuning_metadata
            persist_tuning_metadata(self, tuning_metadata)
        else:
            loaders = captured_loaders.get(id(self.data))
            reve.validate_official_protocol(
                self,
                loaders=loaders,
                n_total_params=self._n_total_params,
                n_trainable_params=self._n_trainable_params,
                allow_target_scaler=target_scaler_mode == "zscore",
            )
        brain_module = getattr(self, "_brain_module", None)
        if robust_loss != "mse":
            if brain_module is None or not isinstance(getattr(brain_module, "loss", None), nn.Module):
                raise RuntimeError("robust loss requires the prepared BrainModule loss")
            brain_module.loss = build_training_loss(robust_loss)
            persist_tuning_metadata(self, {"robust_loss": robust_loss, "smooth_l1_beta": 1.0})
        if target_scaler_mode == "zscore":
            scaler = getattr(brain_module, "target_scaler", None)
            if not isinstance(scaler, TrainingOnlyTargetZScore) or scaler._mean is None:
                raise RuntimeError("target z-score scaler was not fitted on training targets")
            train_subject_ids: list[str] = []
            train_timeline_ids: list[str] = []
            if manifest_path is not None:
                import csv
                with manifest_path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        if row.get("split") == "train":
                            train_subject_ids.append(str(row.get("subject", "")))
                            train_timeline_ids.append(str(row.get("recording_relpath", "")))
            persist_tuning_metadata(
                self,
                {
                    "target_scaler": target_scaler_metadata(
                        scaler=scaler,
                        train_subject_ids=train_subject_ids,
                        train_timeline_ids=train_timeline_ids,
                    )
                },
            )
        if correlation_loss_lambda:
            brain_module = getattr(self, "_brain_module", None)
            if brain_module is None or not isinstance(getattr(brain_module, "loss", None), nn.Module):
                raise RuntimeError("correlation loss requires the prepared BrainModule loss")
            brain_module.loss = CorrelationAuxiliaryLoss(
                brain_module.loss,
                coefficient=correlation_loss_lambda,
            )
            persist_tuning_metadata(
                self,
                {
                    "correlation_loss_lambda": float(correlation_loss_lambda),
                    "correlation_loss_objective": "batch_pearson",
                },
            )
        if augmentation_consistency:
            brain_module = getattr(self, "_brain_module", None)
            _patch_augmentation_consistency_training_step(
                brain_module,
                run_seed=int(getattr(self, "seed", 0)),
                lambda_consistency=augmentation_consistency_lambda,
                noise_scale=augmentation_noise_scale,
                patched_modules=patched_augmentation_consistency_training_steps,
            )
            persist_tuning_metadata(
                self,
                {
                    "augmentation_consistency": True,
                    "augmentation_consistency_lambda": float(augmentation_consistency_lambda),
                    "augmentation_noise_scale": float(augmentation_noise_scale),
                    "augmentation_consistency_batch_size": (
                        AUGMENTATION_CONSISTENCY_BATCH_SIZE
                        if augmentation_consistency
                        else None
                    ),
                    "augmentation_space": "neuro_input",
                    "augmentation_scope": "train_only",
                    "augmentation_pairing": "same_batch_example_two_views",
                },
            )
        return result

    # NeuralBench's CLI and experiment_config modules each keep a local alias
    # to the YAML loader. Patch both so a task-specific grid cannot silently
    # reintroduce the default (33, 34, 35) seed expansion.
    import neuralbench.cli as cli
    import neuralbench.experiment_config as experiment_config

    original_cli_load_yaml_config = cli.load_yaml_config
    original_experiment_load_yaml_config = experiment_config.load_yaml_config

    originals = {
        "data_mode": data_mode,
        "patched_study_source": data_mode == "manifest",
        "patched_selective_info_compat": data_mode == "selective_task",
        "patched_eeglab_reader": _should_use_simplified_eeglab_reader(data_mode),
        "study_class": neuralset_study.Study if neuralset_study is not None else None,
        "study_all_timelines": original_study_all_timelines,
        "selective_eeglab_reader": selective_eeglab_reader,
        "iter_timelines": original_iter_timelines,
        "info": original_info,
        "prepare": original_prepare,
        "test": original_test,
        "setup_run": original_setup_run,
        "prepare_pl_module": original_prepare_pl_module,
        "patched_brain_modules": patched_brain_modules,
        "patched_h6_training_steps": patched_h6_training_steps,
        "patched_augmentation_consistency_training_steps": (
            patched_augmentation_consistency_training_steps
        ),
        "setup_trainer": original_setup_trainer,
        "cli_loader": (cli, original_cli_load_yaml_config),
        "experiment_loader": (
            experiment_config,
            original_experiment_load_yaml_config,
        ),
    }

    def load_seed_grid(path: Path, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name == "grid.yaml":
            return {"seed": list(resolved_seeds)}
        return original_cli_load_yaml_config(path, *args, **kwargs)

    def load_seed_grid_for_experiment_config(
        path: Path, *args: Any, **kwargs: Any
    ) -> Any:
        if Path(path).name == "grid.yaml":
            return {"seed": list(resolved_seeds)}
        return original_experiment_load_yaml_config(path, *args, **kwargs)

    try:
        if data_mode == "manifest":
            assert timelines is not None
            shirazi2024hbn.Shirazi2024Hbn.iter_timelines = iter_manifest_timelines
            if original_info is not None:
                shirazi2024hbn.Shirazi2024Hbn._info = original_info.model_copy(update={"num_timelines": len(timelines)})
        if _should_use_simplified_eeglab_reader(data_mode):
            originals["selective_eeglab_reader"] = _install_selective_eeglab_mat_reader()
        if data_mode == "selective_task":
            assert neuralset_study is not None
            neuralset_study.Study._all_timelines = selective_all_timelines
        Data.prepare = prepare_and_capture
        Experiment._test = test_and_capture
        Experiment.setup_trainer = setup_with_evaluation_callbacks
        Experiment.setup_run = setup_with_metadata
        Experiment.prepare_pl_module = prepare_with_protocol
        cli.load_yaml_config = load_seed_grid
        experiment_config.load_yaml_config = load_seed_grid_for_experiment_config
    except BaseException as active_error:
        try:
            hooks._restore_official_components(originals)
        except BaseException as cleanup_error:
            active_error.add_note(f"patch setup cleanup failure: {cleanup_error!r}")
        raise
    return originals


def _restore_official_components(originals: Mapping[str, Any], *, restore_tuned: Any) -> None:
    from neuralbench.data import Data
    from neuralbench.main import Experiment
    from neuralfetch.studies import shirazi2024hbn

    cli, original_cli_loader = originals["cli_loader"]
    experiment_config, original_experiment_loader = originals["experiment_loader"]
    restoration_errors: list[tuple[str, BaseException]] = []

    def attempt(label: str, restore: Any) -> None:
        try:
            restore()
        except BaseException as error:
            restoration_errors.append((label, error))

    if originals.get("patched_study_source", True):
        attempt("Shirazi2024Hbn.iter_timelines", lambda: setattr(shirazi2024hbn.Shirazi2024Hbn, "iter_timelines", originals["iter_timelines"]))
        attempt("Shirazi2024Hbn._info", lambda: setattr(shirazi2024hbn.Shirazi2024Hbn, "_info", originals["info"]))
    if originals.get("patched_eeglab_reader", False):
        reader = originals.get("selective_eeglab_reader")
        if reader is not None:
            attempt(
                "mne.io.eeglab.eeglab._readmat",
                lambda: setattr(reader["module"], "_readmat", reader["readmat"]),
            )
    if originals.get("patched_selective_info_compat", False):
        attempt(
            "neuralset.Study._all_timelines",
            lambda: setattr(
                originals["study_class"],
                "_all_timelines",
                originals["study_all_timelines"],
            ),
        )
    attempt("Data.prepare", lambda: setattr(Data, "prepare", originals["prepare"]))
    attempt("Experiment.setup_run", lambda: setattr(Experiment, "setup_run", originals["setup_run"]))
    attempt("Experiment._test", lambda: setattr(Experiment, "_test", originals["test"]))
    attempt("Experiment.prepare_pl_module", lambda: setattr(Experiment, "prepare_pl_module", originals["prepare_pl_module"]))
    attempt("Experiment.setup_trainer", lambda: setattr(Experiment, "setup_trainer", originals["setup_trainer"]))
    attempt("neuralbench.cli.load_yaml_config", lambda: setattr(cli, "load_yaml_config", original_cli_loader))
    attempt("neuralbench.experiment_config.load_yaml_config", lambda: setattr(experiment_config, "load_yaml_config", original_experiment_loader))
    attempt("last_tuned.configure_optimizers", lambda: restore_tuned(originals.get("patched_brain_modules", [])))
    attempt("h6.training_step", lambda: _restore_h6_training_steps(originals.get("patched_h6_training_steps", [])))
    attempt(
        "augmentation_consistency.training_step",
        lambda: _restore_augmentation_consistency_training_steps(
            originals.get("patched_augmentation_consistency_training_steps", [])
        ),
    )

    if restoration_errors:
        error = RuntimeError("official component restoration failed")
        for label, restoration_error in restoration_errors:
            error.add_note(f"{label}: {restoration_error!r}")
        raise error


def run_official_subset(
    *,
    manifest_path: Path | None = None,
    data_root: Path,
    epoch_metrics_path: Path,
    selection_path: Path,
    config_path: Path,
    head_variant: str = "mean_linear",
    layer_index: int = -1,
    layer_indices: Sequence[int] | None = None,
    layer_mix_alpha: float = 0.0,
    head_dropout: float = 0.0,
    mean_gradient_scale: float = 0.5,
    correction_gradient_scale: float = 1.0,
    swa_window: int = 0,
    correlation_loss_lambda: float = 0.0,
    robust_loss: str = "mse",
    target_scaler_mode: str = "none",
    seeds: Sequence[int] = (33,),
    evaluation_protocol: str = "strict",
    strict_final_test: bool = False,
    two_stage_finetune: bool = False,
    two_stage_warmup_epochs: int = 3,
    two_stage_unfreeze_last_blocks: int = 1,
    two_stage_encoder_gradient_scale: float = 0.1,
    augmentation_consistency: bool = False,
    augmentation_consistency_lambda: float = 0.0,
    augmentation_noise_scale: float = AUGMENTATION_CONSISTENCY_NOISE_SCALE,
    continued_pretraining: bool = False,
    pretraining_epochs: int = 1,
    pretraining_mask_fraction: float = 0.15,
    pretraining_mask_block_samples: int = 20,
    pretraining_learning_rate: float = 1e-5,
    pretraining_weight_decay: float = 0.05,
    pretraining_max_batches: int = 0,
    data_mode: str = "manifest",
    provenance_path: Path | None = None,
    acquisition_provenance_path: Path | None = None,
    acquisition_provenance_sha256: str | None = None,
    timeline_count: int | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    evidence_recorder: Any | None = None,
    hooks: Any,
) -> list[dict[str, Any]]:
    """Run official REVE on the selected source and collect cached results."""

    os.environ["NEURALBENCH_CONFIG"] = str(config_path)
    final_results: list[dict[str, Any]] = []
    originals: Mapping[str, Any] | None = None
    benchmark_aggregator: Any = None
    original_aggregator_prepare: Any = None
    try:
        originals = hooks._patch_official_components(
            manifest_path,
            data_root,
            epoch_metrics_path,
            selection_path,
            head_variant=head_variant,
            layer_index=layer_index,
            layer_indices=layer_indices,
            layer_mix_alpha=layer_mix_alpha,
            head_dropout=head_dropout,
            mean_gradient_scale=mean_gradient_scale,
            correction_gradient_scale=correction_gradient_scale,
            swa_window=swa_window,
            correlation_loss_lambda=correlation_loss_lambda,
            robust_loss=robust_loss,
            target_scaler_mode=target_scaler_mode,
            seeds=seeds,
            evaluation_protocol=evaluation_protocol,
            strict_final_test=strict_final_test,
            two_stage_finetune=two_stage_finetune,
            two_stage_warmup_epochs=two_stage_warmup_epochs,
            two_stage_unfreeze_last_blocks=two_stage_unfreeze_last_blocks,
            two_stage_encoder_gradient_scale=two_stage_encoder_gradient_scale,
            augmentation_consistency=augmentation_consistency,
            augmentation_consistency_lambda=augmentation_consistency_lambda,
            augmentation_noise_scale=augmentation_noise_scale,
            continued_pretraining=continued_pretraining,
            pretraining_epochs=pretraining_epochs,
            pretraining_mask_fraction=pretraining_mask_fraction,
            pretraining_mask_block_samples=pretraining_mask_block_samples,
            pretraining_learning_rate=pretraining_learning_rate,
            pretraining_weight_decay=pretraining_weight_decay,
            pretraining_max_batches=pretraining_max_batches,
            data_mode=data_mode,
            provenance_path=provenance_path,
            acquisition_provenance_path=acquisition_provenance_path,
            acquisition_provenance_sha256=acquisition_provenance_sha256,
            timeline_count=timeline_count,
            run_metadata=run_metadata,
            final_results=final_results,
            evidence_recorder=evidence_recorder,
        )
        from neuralbench.main import BenchmarkAggregator

        benchmark_aggregator = BenchmarkAggregator
        original_aggregator_prepare = BenchmarkAggregator.prepare
        BenchmarkAggregator.prepare = (
            lambda aggregator: _run_experiments_synchronously(
                aggregator,
                evidence_recorder=evidence_recorder,
            )
        )
        from neuralbench import run_benchmark

        run_benchmark(device="eeg", task="age", model="reve", force=True)
        # The public runner returns no result for a non-debug local run. The
        # official Experiment._test result is captured above instead. Avoid a
        # second ``plot_cached`` call: it reconstructs the canonical mean head
        # and collides with this run's custom upstream head UID.
        return final_results
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[BaseException] = []
        if active_error is not None and data_mode in {"full", "selective_task"} and provenance_path is not None:
            try:
                provenance_path.unlink(missing_ok=True)
            except BaseException as error:
                cleanup_errors.append(error)
        if benchmark_aggregator is not None and original_aggregator_prepare is not None:
            try:
                benchmark_aggregator.prepare = original_aggregator_prepare
            except BaseException as error:
                cleanup_errors.append(error)
        if originals is not None:
            try:
                hooks._restore_official_components(originals)
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if active_error is not None:
                for cleanup_error in cleanup_errors:
                    active_error.add_note(f"cleanup failure: {cleanup_error!r}")
            else:
                cleanup_error = RuntimeError("official REVE cleanup failed")
                for error in cleanup_errors:
                    cleanup_error.add_note(repr(error))
                raise cleanup_error


def _run_experiments_synchronously(
    aggregator: Any,
    *,
    evidence_recorder: Any | None = None,
) -> None:
    """Run prepared NeuralBench experiments in the current process.

    NeuralBench's public non-debug runner submits experiments to an ``exca``
    job array and returns before those workers finish when no scheduler is
    available. The fixed-manifest harness needs the worker-local monkeypatches
    above to survive into the actual experiment, so execute each prepared
    experiment directly while retaining the canonical (non-debug) config.
    """

    for experiment in aggregator.experiments:
        if evidence_recorder is None:
            experiment.run()
        else:
            with evidence_recorder.phase("fit_and_evaluation"):
                experiment.run()
