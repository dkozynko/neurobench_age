"""Controlled two-stage fine-tuning for the official REVE age runner."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from lightning.pytorch.callbacks import Callback


_TRANSFORMER_LAYER_RE = re.compile(r"(?:^|\.)transformer\.layers\.(\d+)$")


@dataclass(frozen=True)
class TwoStageFineTuneConfig:
    """Bounded schedule for the two-stage encoder adaptation experiment."""

    warmup_epochs: int = 3
    unfreeze_last_blocks: int = 1
    encoder_gradient_scale: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.warmup_epochs, bool) or self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a non-negative integer")
        if isinstance(self.unfreeze_last_blocks, bool) or self.unfreeze_last_blocks < 1:
            raise ValueError("unfreeze_last_blocks must be a positive integer")
        if not 0.0 < float(self.encoder_gradient_scale) <= 1.0:
            raise ValueError("encoder_gradient_scale must be in (0, 1]")


def validate_two_stage_options(
    *,
    head_variant: str,
    data_mode: str,
    evaluation_protocol: str,
) -> None:
    """Keep this experiment comparable to the declared 1000-subject screen."""

    if head_variant not in {
        "mean_linear",
        "mean_linear_copy",
        "mean_rich_stats_residual",
        "multi_query_rich_stats",
    }:
        raise ValueError(
            "two-stage fine-tuning currently requires mean_linear, "
            "mean_linear_copy, mean_rich_stats_residual, or multi_query_rich_stats"
        )
    if data_mode != "manifest":
        raise ValueError("two-stage fine-tuning currently requires manifest data mode")
    if evaluation_protocol != "strict":
        raise ValueError("two-stage fine-tuning currently requires strict evaluation")


def _model_encoder(model: Any) -> tuple[Any, str]:
    direct_encoder = getattr(model, "encoder", None)
    if direct_encoder is not None and callable(
        getattr(direct_encoder, "named_parameters", None)
    ):
        # UpstreamReveHeadModel exposes the pretrained encoder and custom
        # prediction head as direct sibling modules.
        return direct_encoder, "encoder"
    wrapped = getattr(model, "wrapped_model", None)
    if wrapped is not None and callable(getattr(wrapped, "named_parameters", None)):
        nested_encoder = getattr(wrapped, "encoder", None)
        if nested_encoder is not None and callable(
            getattr(nested_encoder, "named_parameters", None)
        ):
            # Custom upstream heads are returned as the wrapped model by
            # NeuralBench. Their REVE encoder and prediction head are sibling
            # modules, so the whole wrapped model cannot be treated as the
            # encoder without hiding the head parameters from the plan.
            return nested_encoder, "wrapped_model.encoder"
        return wrapped, "wrapped_model"
    if callable(getattr(model, "named_parameters", None)):
        return model, ""
    raise TypeError("two-stage fine-tuning requires a model with named_parameters()")


def inspect_two_stage_plan(
    model: Any,
    *,
    unfreeze_last_blocks: int = 1,
) -> dict[str, Any]:
    """Resolve transformer block prefixes and parameter counts before training."""

    if isinstance(unfreeze_last_blocks, bool) or unfreeze_last_blocks < 1:
        raise ValueError("unfreeze_last_blocks must be a positive integer")
    encoder, encoder_root = _model_encoder(model)
    layer_prefixes: list[tuple[int, str]] = []
    for name, _module in encoder.named_modules():
        match = _TRANSFORMER_LAYER_RE.search(name)
        if match is not None:
            prefix = ".".join(part for part in (encoder_root, name) if part)
            layer_prefixes.append((int(match.group(1)), prefix))
    if not layer_prefixes:
        raise ValueError(
            "two-stage fine-tuning could not find transformer.layers.N in the encoder"
        )
    layer_prefixes.sort()
    if unfreeze_last_blocks > len(layer_prefixes):
        raise ValueError(
            f"requested {unfreeze_last_blocks} final blocks, but encoder exposes "
            f"only {len(layer_prefixes)}"
        )

    selected_layers = layer_prefixes[-unfreeze_last_blocks:]
    parameter_items = list(model.named_parameters())
    encoder_prefix = f"{encoder_root}." if encoder_root else ""
    encoder_names = [
        name for name, _parameter in parameter_items if name.startswith(encoder_prefix)
    ]
    last_block_names = [
        name
        for name, _parameter in parameter_items
        if any(name.startswith(prefix + ".") for _index, prefix in selected_layers)
    ]
    if not last_block_names:
        raise ValueError("selected transformer blocks expose no parameters")
    head_names = [name for name, _parameter in parameter_items if name not in encoder_names]
    if not head_names:
        raise ValueError("two-stage fine-tuning found no downstream head parameters")
    return {
        "encoder_root": encoder_root or None,
        "transformer_layer_indices": [index for index, _prefix in layer_prefixes],
        "selected_layer_indices": [index for index, _prefix in selected_layers],
        "last_layer_prefix": selected_layers[-1][1],
        "last_layer_prefixes": [prefix for _index, prefix in selected_layers],
        "encoder_parameter_names": encoder_names,
        "last_block_parameter_names": last_block_names,
        "head_parameter_names": head_names,
        "encoder_parameter_count": len(encoder_names),
        "last_block_parameter_count": len(last_block_names),
        "head_parameter_count": len(head_names),
    }


class TwoStageFineTuneCallback(Callback):
    """Freeze the encoder for warm-up, then enable only final blocks."""

    def __init__(
        self,
        config: TwoStageFineTuneConfig,
        *,
        metadata_path: Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.metadata_path = Path(metadata_path)
        self.stage = "not_started"
        self.plan: dict[str, Any] | None = None
        self._parameter_by_name: dict[str, Any] = {}
        self._stage_events: list[dict[str, Any]] = []

    @staticmethod
    def _trainable_names(model: Any) -> list[str]:
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    def _write_metadata(self) -> None:
        if self.plan is None:
            return
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "config": asdict(self.config),
            "stage": self.stage,
            "plan": self.plan,
            "stage_events": list(self._stage_events),
        }
        self.metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _record_stage(self, model: Any, *, epoch: int, stage: str) -> None:
        self.stage = stage
        self._stage_events.append(
            {
                "epoch": int(epoch),
                "stage": stage,
                "trainable_parameter_names": self._trainable_names(model),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            }
        )
        self._write_metadata()

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer
        model = getattr(pl_module, "model", None)
        if model is None:
            raise RuntimeError("two-stage fine-tuning requires pl_module.model")
        self.plan = inspect_two_stage_plan(
            model,
            unfreeze_last_blocks=self.config.unfreeze_last_blocks,
        )
        self._parameter_by_name = dict(model.named_parameters())
        encoder_names = set(self.plan["encoder_parameter_names"])
        for name, parameter in self._parameter_by_name.items():
            parameter.requires_grad = name not in encoder_names
        self._record_stage(model, epoch=0, stage="head_warmup")

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer
        if self.plan is None:
            raise RuntimeError("two-stage fine-tuning was not initialized")
        if self.stage != "head_warmup":
            return
        epoch = int(getattr(pl_module, "current_epoch", 0))
        if epoch < self.config.warmup_epochs:
            return
        selected_names = set(self.plan["last_block_parameter_names"])
        for name, parameter in self._parameter_by_name.items():
            parameter.requires_grad = name in selected_names or name in set(
                self.plan["head_parameter_names"]
            )
        self._record_stage(model=pl_module.model, epoch=epoch, stage="last_block_adaptation")

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        if self.stage != "last_block_adaptation" or self.plan is None:
            return
        for name in self.plan["last_block_parameter_names"]:
            parameter = self._parameter_by_name[name]
            if parameter.grad is not None:
                parameter.grad.mul_(self.config.encoder_gradient_scale)


__all__ = [
    "TwoStageFineTuneCallback",
    "TwoStageFineTuneConfig",
    "inspect_two_stage_plan",
    "validate_two_stage_options",
]
