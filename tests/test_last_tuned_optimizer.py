from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import neurobench_age.heads.upstream as reve


class _TinyLastTunedModel(nn.Module):
    """Small backbone/head topology used to exercise optimizer identity rules."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
        self.head = reve.UpstreamReveHead(variant="last_tuned", embed_dim=4, n_outputs=1, dropout=0.0, query_token=torch.full((1, 1, 4), 0.25))


class _DownstreamWrapperShape(nn.Module):
    """NeuralBench's production wrapper shape around a tuned model."""

    def __init__(self) -> None:
        super().__init__()
        self.wrapped_model = _TinyLastTunedModel()
        self.aggregation = nn.Identity()
        self.probe = nn.Identity()


def _build_configuration(
    model: nn.Module, *, estimated_stepping_batches: int = 37
) -> dict[str, object]:
    """Invoke the narrowly-scoped tuned optimizer seam under test."""

    trainer = SimpleNamespace(estimated_stepping_batches=estimated_stepping_batches)
    return reve.build_last_tuned_optimizer_config(model, trainer=trainer)


def _parameter_names(model: nn.Module, parameters: list[nn.Parameter]) -> list[str]:
    names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    return [names_by_identity[id(parameter)] for parameter in parameters]


def test_last_tuned_optimizer_has_exact_ordered_base_and_query_groups() -> None:
    model = _TinyLastTunedModel()

    configuration = _build_configuration(model)
    optimizer = configuration["optimizer"]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2

    base_group, query_group = optimizer.param_groups
    assert base_group["max_lr"] == pytest.approx(1e-4)
    assert base_group["initial_lr"] == pytest.approx(4e-6)
    assert base_group["lr"] == pytest.approx(4e-6)
    assert base_group["weight_decay"] == pytest.approx(0.05)
    assert query_group["max_lr"] == pytest.approx(1e-5)
    assert query_group["initial_lr"] == pytest.approx(4e-7)
    assert query_group["lr"] == pytest.approx(4e-7)
    assert query_group["weight_decay"] == pytest.approx(0.05)

    base_parameters = list(base_group["params"])
    query_parameters = list(query_group["params"])
    all_trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    assert _parameter_names(model, base_parameters) == [
        "backbone.0.weight",
        "backbone.0.bias",
        "backbone.1.weight",
        "backbone.1.bias",
        "head.gate_logit",
        "head.norm.weight",
        "head.linear.weight",
        "head.linear.bias",
    ]
    assert [id(parameter) for parameter in query_parameters] == [
        id(model.head.query_token)
    ]
    assert _parameter_names(model, base_parameters + query_parameters) == [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter is not model.head.query_token
    ] + ["head.query_token"]
    assert len({id(parameter) for parameter in base_parameters + query_parameters}) == len(all_trainable)
    assert {id(parameter) for parameter in base_parameters + query_parameters} == {
        id(parameter) for parameter in all_trainable
    }


def test_last_tuned_one_cycle_preserves_differential_lr_schedule_invariants() -> None:
    total_steps = 37
    configuration = _build_configuration(_TinyLastTunedModel(), estimated_stepping_batches=total_steps)
    optimizer = configuration["optimizer"]
    scheduler_config = configuration["lr_scheduler"]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler_config, dict)
    assert scheduler_config["interval"] == "step"
    assert scheduler_config["frequency"] == 1

    scheduler = scheduler_config["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
    assert scheduler.total_steps == total_steps
    assert scheduler._anneal_func_type == "cos"
    assert scheduler._schedule_phases[0]["end_step"] == pytest.approx(total_steps * 0.1 - 1)
    assert [group["initial_lr"] for group in optimizer.param_groups] == pytest.approx([4e-6, 4e-7])
    assert [group["max_lr"] for group in optimizer.param_groups] == pytest.approx([1e-4, 1e-5])
    assert [group["min_lr"] for group in optimizer.param_groups] == pytest.approx([4e-10, 4e-11])
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([4e-6, 4e-7])


def test_last_tuned_optimizer_resolves_neuralbench_downstream_wrapper_shape() -> None:
    wrapper = _DownstreamWrapperShape()
    configuration = _build_configuration(wrapper)
    optimizer = configuration["optimizer"]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert [id(parameter) for parameter in optimizer.param_groups[1]["params"]] == [
        id(wrapper.wrapped_model.head.query_token)
    ]
    metadata = reve.last_tuned_optimizer_metadata(wrapper)
    assert metadata["param_groups"][1]["parameter_names"] == ["head.query_token"]
