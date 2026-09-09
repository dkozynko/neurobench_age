from __future__ import annotations

import pytest
import torch
from torch import nn

from neurobench_age.heads.math import (
    MeanLinearCopyHead,
    MeanMLPResidualHead,
    MeanRichStatsResidualHead,
)
from neurobench_age.pipelines.capacity_data_regime import (
    CAPACITY_HEADS,
    build_capacity_data_regime_head,
    capacity_head_complexity_metadata,
)
from neurobench_age.pipelines.frozen_probe_training import APPROVED_HEADS


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_extension_allowlist_builds_exact_matched_head_architecture() -> None:
    assert CAPACITY_HEADS == (
        "mean_linear",
        "mean_rich_stats_residual",
        "mean_mlp_residual_matched(hidden_dim=4)",
    )
    head = build_capacity_data_regime_head(
        "mean_mlp_residual_matched(hidden_dim=4)", embed_dim=512, n_outputs=1
    )

    assert isinstance(head, MeanMLPResidualHead)
    assert head.embed_dim == 512
    assert head.hidden_dim == 4
    assert isinstance(head.hidden, nn.Linear)
    assert head.hidden.bias is not None
    assert isinstance(head.correction, nn.Linear)
    assert head.correction.bias is None
    assert torch.equal(head.correction.weight, torch.zeros_like(head.correction.weight))
    assert _parameter_count(head) == 2_569

    tokens = torch.randn(2, 5, 512)
    baseline = MeanLinearCopyHead(embed_dim=512, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)
    torch.testing.assert_close(head.pool_tokens(tokens), tokens.mean(dim=1))
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_extension_head_complexity_records_exact_counts_and_operations() -> None:
    rich = build_capacity_data_regime_head(
        "mean_rich_stats_residual", embed_dim=512, n_outputs=1
    )
    assert isinstance(rich, MeanRichStatsResidualHead)
    assert _parameter_count(rich) == 2_561

    metadata = capacity_head_complexity_metadata(
        "mean_mlp_residual_matched(hidden_dim=4)", embed_dim=512, n_outputs=1
    )
    assert metadata["parameter_count"] == 2_569
    assert metadata["hidden_dim"] == 4
    assert metadata["operations"] == [
        "mean_pool",
        "baseline_linear",
        "hidden_linear",
        "gelu",
        "correction_linear",
    ]


@pytest.mark.parametrize(
    "head_name",
    [
        "mean_layer_linear",
        "mean_mlp_residual",
        "mean_mlp_residual_matched",
        "unknown",
    ],
)
def test_extension_builder_rejects_non_allowlisted_heads(head_name: str) -> None:
    with pytest.raises(ValueError, match="exact extension allowlist"):
        build_capacity_data_regime_head(head_name, embed_dim=512, n_outputs=1)


def test_extension_head_does_not_mutate_primary_registry() -> None:
    assert "mean_mlp_residual_matched(hidden_dim=4)" not in APPROVED_HEADS
