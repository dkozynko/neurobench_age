from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neurobench_age.training.two_stage import (
    TwoStageFineTuneConfig,
    TwoStageFineTuneCallback,
    inspect_two_stage_plan,
    validate_two_stage_options,
)


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([_TinyBlock(), _TinyBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer.layers:
            x = layer(x)
        return x


class _TinyDownstream(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wrapped_model = _TinyEncoder()
        self.probe = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(self.wrapped_model(x))


class _TinyNestedDownstream(nn.Module):
    """Mirror NeuralTrain's _ReveWrapper.model.transformer.layers naming."""

    def __init__(self) -> None:
        super().__init__()
        self.wrapped_model = nn.Module()
        self.wrapped_model.model = _TinyEncoder()
        self.probe = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(self.wrapped_model.model(x))


class _TinyNestedHeadDownstream(nn.Module):
    """Mirror the custom upstream-head model nested in NeuralBench's wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.wrapped_model = nn.Module()
        self.wrapped_model.encoder = _TinyEncoder()
        self.wrapped_model.head = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrapped_model.head(self.wrapped_model.encoder(x).mean(dim=1))


class _TinyDirectHeadDownstream(nn.Module):
    """Mirror UpstreamReveHeadModel's direct encoder/head layout."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TinyEncoder()
        self.head = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).mean(dim=1))


def test_inspect_plan_selects_only_last_transformer_block() -> None:
    plan = inspect_two_stage_plan(_TinyDownstream(), unfreeze_last_blocks=1)

    assert plan["last_layer_prefix"] == "wrapped_model.transformer.layers.1"
    assert plan["encoder_parameter_count"] > 0
    assert plan["last_block_parameter_names"]
    assert all(
        name.startswith("wrapped_model.transformer.layers.1.")
        for name in plan["last_block_parameter_names"]
    )


def test_inspect_plan_handles_nested_reve_wrapper_transformer_blocks() -> None:
    plan = inspect_two_stage_plan(_TinyNestedDownstream(), unfreeze_last_blocks=1)

    assert plan["last_layer_prefix"] == "wrapped_model.model.transformer.layers.1"
    assert all(
        name.startswith("wrapped_model.model.transformer.layers.1.")
        for name in plan["last_block_parameter_names"]
    )


def test_inspect_plan_keeps_nested_custom_head_outside_encoder() -> None:
    plan = inspect_two_stage_plan(_TinyNestedHeadDownstream(), unfreeze_last_blocks=1)

    assert plan["encoder_root"] == "wrapped_model.encoder"
    assert plan["head_parameter_names"] == [
        "wrapped_model.head.weight",
        "wrapped_model.head.bias",
    ]


def test_inspect_plan_keeps_direct_custom_head_outside_encoder() -> None:
    plan = inspect_two_stage_plan(_TinyDirectHeadDownstream(), unfreeze_last_blocks=1)

    assert plan["encoder_root"] == "encoder"
    assert plan["head_parameter_names"] == ["head.weight", "head.bias"]


def test_callback_freezes_encoder_then_unfreezes_last_block(tmp_path) -> None:
    model = _TinyDownstream()
    callback = TwoStageFineTuneCallback(
        TwoStageFineTuneConfig(warmup_epochs=2, encoder_gradient_scale=0.1),
        metadata_path=tmp_path / "two_stage.json",
    )
    module = SimpleNamespace(model=model, current_epoch=0)

    callback.on_fit_start(SimpleNamespace(), module)

    assert all(not parameter.requires_grad for parameter in model.wrapped_model.parameters())
    assert all(parameter.requires_grad for parameter in model.probe.parameters())
    assert callback.stage == "head_warmup"

    module.current_epoch = 2
    callback.on_train_epoch_start(SimpleNamespace(), module)

    assert callback.stage == "last_block_adaptation"
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("wrapped_model.transformer.layers.1.")
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("wrapped_model.transformer.layers.0.")
    )
    assert (tmp_path / "two_stage.json").is_file()


def test_callback_scales_only_last_block_gradients_after_transition(tmp_path) -> None:
    model = _TinyDownstream()
    callback = TwoStageFineTuneCallback(
        TwoStageFineTuneConfig(warmup_epochs=0, encoder_gradient_scale=0.1),
        metadata_path=tmp_path / "two_stage.json",
    )
    module = SimpleNamespace(model=model, current_epoch=0)
    callback.on_fit_start(SimpleNamespace(), module)
    callback.on_train_epoch_start(SimpleNamespace(), module)

    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    before = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    callback.on_after_backward(SimpleNamespace(), module)

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("wrapped_model.transformer.layers.1."):
            torch.testing.assert_close(parameter.grad, 0.1 * before[name])
        else:
            torch.testing.assert_close(parameter.grad, before[name])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup_epochs": -1}, "warmup_epochs"),
        ({"unfreeze_last_blocks": 0}, "unfreeze_last_blocks"),
        ({"encoder_gradient_scale": 0.0}, "encoder_gradient_scale"),
    ],
)
def test_config_rejects_invalid_bounds(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TwoStageFineTuneConfig(**kwargs)


def test_two_stage_options_allow_matched_rich_stats_screen() -> None:
    validate_two_stage_options(
        head_variant="mean_linear",
        data_mode="manifest",
        evaluation_protocol="strict",
    )

    validate_two_stage_options(
        head_variant="mean_linear_copy",
        data_mode="manifest",
        evaluation_protocol="strict",
    )

    validate_two_stage_options(
        head_variant="mean_rich_stats_residual",
        data_mode="manifest",
        evaluation_protocol="strict",
    )


def test_two_stage_options_allow_multi_query_rich_stats_screen() -> None:
    validate_two_stage_options(
        head_variant="multi_query_rich_stats",
        data_mode="manifest",
        evaluation_protocol="strict",
    )

    with pytest.raises(
        ValueError,
        match="mean_linear, mean_linear_copy, mean_rich_stats_residual, or multi_query_rich_stats",
    ):
        validate_two_stage_options(
            head_variant="mean_mlp_residual",
            data_mode="manifest",
            evaluation_protocol="strict",
        )
    with pytest.raises(ValueError, match="manifest"):
        validate_two_stage_options(
            head_variant="mean_linear",
            data_mode="selective_task",
            evaluation_protocol="strict",
        )
    with pytest.raises(ValueError, match="strict"):
        validate_two_stage_options(
            head_variant="mean_linear",
            data_mode="manifest",
            evaluation_protocol="legacy",
        )
