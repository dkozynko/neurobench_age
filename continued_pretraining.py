"""Train-only masked self-distillation for the REVE EEG encoder.

The routine in this module is deliberately independent of age labels.  It
uses the encoder's clean embedding as a stop-gradient teacher and asks the
same encoder to reconstruct that embedding from an input with contiguous
time blocks masked out.  It is intended to run once, before the official
supervised Lightning fit starts.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

try:
    from lightning.pytorch.callbacks import Callback as LightningCallback
except ImportError:  # Keep pure helpers importable without Lightning.
    class LightningCallback:  # type: ignore[no-redef]
        """Fallback base used only when the official stack is absent."""


@dataclass(frozen=True)
class ContinuedPretrainingConfig:
    """Bounded configuration for train-only continued encoder pretraining."""

    epochs: int = 1
    mask_fraction: float = 0.15
    mask_block_samples: int = 20
    learning_rate: float = 1e-5
    weight_decay: float = 0.05
    max_batches: int | None = None
    gradient_clip_val: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("pretraining epochs must be a positive integer")
        if not math.isfinite(float(self.mask_fraction)) or not 0.0 < float(self.mask_fraction) <= 1.0:
            raise ValueError("pretraining mask_fraction must be finite and in (0, 1]")
        if (
            isinstance(self.mask_block_samples, bool)
            or not isinstance(self.mask_block_samples, int)
            or self.mask_block_samples < 1
        ):
            raise ValueError("pretraining mask_block_samples must be a positive integer")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("gradient_clip_val", self.gradient_clip_val),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"pretraining {name} must be finite and non-negative")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("pretraining learning_rate must be positive")
        if self.max_batches is not None and (
            isinstance(self.max_batches, bool)
            or not isinstance(self.max_batches, int)
            or self.max_batches < 1
        ):
            raise ValueError("pretraining max_batches must be a positive integer or None")


def _validate_neuro_input(neuro: torch.Tensor) -> None:
    if not isinstance(neuro, torch.Tensor):
        raise TypeError("neuro input must be a torch.Tensor")
    if neuro.ndim != 3:
        raise ValueError(f"neuro input must have shape [B, C, T], got {tuple(neuro.shape)}")
    if not (neuro.is_floating_point() or neuro.is_complex()):
        raise TypeError("neuro input must be floating point")
    if not torch.isfinite(neuro).all():
        raise ValueError("neuro input contains non-finite values")


def _mask_seed(*, run_seed: int, epoch: int, batch_idx: int) -> int:
    # Keep the seed deterministic across Python processes and independent of
    # the global RNG used by Lightning or the official data loader.
    modulus = 2**63 - 1
    return int(
        (
            int(run_seed) * 1_000_003
            + int(epoch) * 9_176
            + int(batch_idx) * 65_537
        )
        % modulus
    )


def mask_neuro_input(
    neuro: torch.Tensor,
    *,
    run_seed: int,
    epoch: int,
    batch_idx: int,
    mask_fraction: float = 0.15,
    mask_block_samples: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask deterministic contiguous time blocks, shared across channels.

    Returns a new tensor and a boolean mask of shape ``[B, 1, T]``.  The
    circular block construction makes the selected spans contiguous even
    when a span crosses the end of the two-second window.
    """

    _validate_neuro_input(neuro)
    if not math.isfinite(float(mask_fraction)) or not 0.0 < float(mask_fraction) <= 1.0:
        raise ValueError("mask_fraction must be finite and in (0, 1]")
    if (
        isinstance(mask_block_samples, bool)
        or not isinstance(mask_block_samples, int)
        or mask_block_samples < 1
    ):
        raise ValueError("mask_block_samples must be a positive integer")

    batch_size, _channels, time_samples = neuro.shape
    target_samples = max(1, int(math.ceil(float(mask_fraction) * time_samples)))
    block_count = max(1, int(math.ceil(target_samples / mask_block_samples)))
    generator = torch.Generator(device=neuro.device)
    generator.manual_seed(_mask_seed(run_seed=run_seed, epoch=epoch, batch_idx=batch_idx))
    starts = torch.randint(
        low=0,
        high=time_samples,
        size=(batch_size, block_count),
        generator=generator,
        device=neuro.device,
    )
    offsets = torch.arange(
        mask_block_samples,
        device=neuro.device,
        dtype=starts.dtype,
    )
    indices = (starts.unsqueeze(-1) + offsets) % time_samples
    mask = torch.zeros(
        batch_size,
        time_samples,
        dtype=torch.bool,
        device=neuro.device,
    )
    mask.scatter_(dim=1, index=indices.reshape(batch_size, -1), src=torch.ones_like(indices, dtype=torch.bool).reshape(batch_size, -1))
    mask = mask.unsqueeze(1)
    masked = neuro.masked_fill(mask.expand_as(neuro), 0.0)
    return masked, mask


def continued_pretraining_loss(
    teacher_embedding: torch.Tensor,
    student_embedding: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE to a detached clean-embedding teacher."""

    if not isinstance(teacher_embedding, torch.Tensor) or not isinstance(student_embedding, torch.Tensor):
        raise TypeError("teacher and student embeddings must be tensors")
    if teacher_embedding.shape != student_embedding.shape:
        raise ValueError(
            "teacher and student embeddings must have equal shapes: "
            f"{tuple(teacher_embedding.shape)} != {tuple(student_embedding.shape)}"
        )
    if not torch.isfinite(teacher_embedding).all() or not torch.isfinite(student_embedding).all():
        raise ValueError("teacher or student embedding contains non-finite values")
    return F.mse_loss(student_embedding, teacher_embedding.detach())


def _encoder_parameters(module: Any) -> list[torch.Tensor]:
    model = getattr(module, "model", None)
    wrapped = getattr(model, "wrapped_model", None)
    candidate = wrapped if wrapped is not None else module
    parameters = [
        parameter
        for parameter in candidate.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("continued pretraining found no trainable encoder parameters")
    return parameters


def _module_device(module: Any, parameters: list[torch.Tensor]) -> torch.device:
    device = getattr(module, "device", None)
    if isinstance(device, torch.device):
        return device
    return parameters[0].device


def _prepare_batch(batch: Any, *, trainer: Any | None, device: torch.device) -> Any:
    if trainer is not None:
        strategy = getattr(trainer, "strategy", None)
        batch_to_device = getattr(strategy, "batch_to_device", None)
        if callable(batch_to_device):
            batch = batch_to_device(batch, device)
    return batch


def _masked_batch(batch: Any, masked_neuro: torch.Tensor) -> Any:
    # NeuralBench batches expose a Mapping-like ``data`` attribute.  Preserve
    # the original object and all fields, replacing only the EEG tensor.
    copied = copy.copy(batch)
    data = getattr(batch, "data", None)
    if not isinstance(data, Mapping):
        raise TypeError("continued pretraining expects batch.data to be a mapping")
    copied.data = dict(data)
    copied.data["neuro"] = masked_neuro
    return copied


def _clean_teacher_embedding(module: Any, batch: Any) -> torch.Tensor:
    """Compute the clean target without dropout or running-stat updates."""

    previous_training = bool(getattr(module, "training", True))
    module.eval()
    try:
        with torch.no_grad():
            teacher = module.model_forward_embedding(batch)
    finally:
        module.train(previous_training)
    if not isinstance(teacher, torch.Tensor):
        raise TypeError("continued pretraining teacher embedding must be a tensor")
    return teacher.detach()


def _reset_train_loader_after_pretraining(train_loader: Any, trainer: Any | None) -> None:
    """Release the consumed iterator before Lightning builds its fit iterator."""

    iterator = getattr(train_loader, "_iterator", None)
    if iterator is not None:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        # PyTorch's persistent-worker DataLoader reuses this private iterator
        # on the next __iter__ call.  Clearing it ensures Lightning starts a
        # fresh epoch rather than observing the exhausted pretraining stream.
        try:
            train_loader._iterator = None
        except (AttributeError, TypeError):
            pass

    # Be defensive for Lightning versions that materialize the fit fetcher
    # before callback hooks run.  The current version initializes these fields
    # after on_fit_start, but clearing them is harmless when they are empty.
    fit_loop = getattr(trainer, "fit_loop", None) if trainer is not None else None
    if fit_loop is not None:
        if getattr(fit_loop, "_data_fetcher", None) is not None:
            teardown = getattr(fit_loop._data_fetcher, "teardown", None)
            if callable(teardown):
                teardown()
        fit_loop._data_fetcher = None
        fit_loop._combined_loader = None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_continued_pretraining(
    module: Any,
    train_loader: Iterable[Any],
    *,
    config: ContinuedPretrainingConfig,
    run_seed: int,
    metadata_path: Path,
    metrics_path: Path,
    trainer: Any | None = None,
) -> dict[str, Any]:
    """Run masked teacher-student pretraining over the train loader only."""

    if not isinstance(config, ContinuedPretrainingConfig):
        raise TypeError("config must be ContinuedPretrainingConfig")
    parameters = _encoder_parameters(module)
    device = _module_device(module, parameters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    module_was_training = bool(getattr(module, "training", True))
    module.train()
    total_batches = 0
    epoch_summaries: list[dict[str, Any]] = []
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, config.epochs + 1):
            losses: list[float] = []
            for batch_idx, original_batch in enumerate(train_loader):
                if config.max_batches is not None and batch_idx >= config.max_batches:
                    break
                batch = _prepare_batch(original_batch, trainer=trainer, device=device)
                data = getattr(batch, "data", None)
                if not isinstance(data, Mapping) or "neuro" not in data:
                    raise TypeError("continued pretraining requires batch.data['neuro']")
                neuro = data["neuro"]
                _validate_neuro_input(neuro)
                teacher = _clean_teacher_embedding(module, batch)
                masked_neuro, _mask = mask_neuro_input(
                    neuro,
                    run_seed=run_seed,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    mask_fraction=config.mask_fraction,
                    mask_block_samples=config.mask_block_samples,
                )
                student = module.model_forward_embedding(_masked_batch(batch, masked_neuro))
                loss = continued_pretraining_loss(teacher, student)
                if not torch.isfinite(loss):
                    raise FloatingPointError("continued pretraining produced a non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.gradient_clip_val > 0.0:
                    torch.nn.utils.clip_grad_norm_(parameters, float(config.gradient_clip_val))
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                total_batches += 1
            if not losses:
                raise RuntimeError(f"continued pretraining epoch {epoch} processed no batches")
            summary = {
                "epoch": epoch,
                "batches": len(losses),
                "loss": sum(losses) / len(losses),
            }
            epoch_summaries.append(summary)
            metrics_file.write(json.dumps(summary, sort_keys=True) + "\n")
            metrics_file.flush()
    _reset_train_loader_after_pretraining(train_loader, trainer)
    if not module_was_training:
        module.eval()
    metadata = {
        "schema_version": 1,
        "config": asdict(config),
        "epochs": config.epochs,
        "batches": total_batches,
        "mask_fraction": config.mask_fraction,
        "mask_block_samples": config.mask_block_samples,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_batches": config.max_batches,
        "age_labels_used": False,
        "source_split": "train_only",
        "validation_test_access": "withheld",
        "objective": "masked_teacher_student_embedding_mse",
        "epoch_summaries": epoch_summaries,
    }
    _write_json(metadata_path, metadata)
    return {
        "epochs": config.epochs,
        "batches": total_batches,
        "age_labels_used": False,
        "source_split": "train_only",
        "validation_test_access": "withheld",
        "objective": "masked_teacher_student_embedding_mse",
        "last_loss": epoch_summaries[-1]["loss"],
    }


class ContinuedPretrainingCallback(LightningCallback):
    """Run continued pretraining exactly once at the start of supervised fit."""

    def __init__(
        self,
        config: ContinuedPretrainingConfig,
        *,
        train_loader: Iterable[Any],
        metadata_path: Path,
        metrics_path: Path,
        run_seed: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.train_loader = train_loader
        self.metadata_path = Path(metadata_path)
        self.metrics_path = Path(metrics_path)
        self.run_seed = int(run_seed)
        self.summary: dict[str, Any] | None = None
        self._ran = False

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        if self._ran:
            return
        self.summary = run_continued_pretraining(
            pl_module,
            self.train_loader,
            config=self.config,
            run_seed=self.run_seed,
            metadata_path=self.metadata_path,
            metrics_path=self.metrics_path,
            trainer=trainer,
        )
        self._ran = True


__all__ = [
    "ContinuedPretrainingCallback",
    "ContinuedPretrainingConfig",
    "continued_pretraining_loss",
    "mask_neuro_input",
    "run_continued_pretraining",
]
