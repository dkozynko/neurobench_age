from __future__ import annotations

import json
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import neurobench_age.heads.upstream as reve

from neurobench_age.heads.upstream import (
    AdapterContractError,
    MeanAnchorEnsembleHead,
    MeanReliabilityShrinkageHead,
    audit_reliability_gates,
    MeanAnchorHead,
    MeanLinearCopyHead,
    MeanLayerLinearHead,
    MeanLayerMixHead,
    MeanLayerMixFixedHead,
    MeanLinearDetachedHead,
    MeanLinearWarmupHead,
    MeanLinearGradientScaledHead,
    MeanLinearProbeScaledHead,
    MeanResidualHead,
    MeanVectorAnchorHead,
    MeanMLPResidualHead,
    MeanStatsResidualHead,
    MeanStatsResidualDetachedHead,
    MeanStatsResidualGradientScaledHead,
    MeanStatsProbeScaledHead,
    MeanStatsAttentionResidualHead,
    MeanAttentionGatedHead,
    GlobalStatsResidualHead,
    MeanRichStatsResidualHead,
    MeanRichStatsGradientRoutesHead,
    MultiQueryRichStatsResidualHead,
    GroupedRichStatsShrinkageHead,
    GroupedStatsSharedGateHead,
    TemporalPyramidStatsResidualHead,
    MeanCovarianceResidualHead,
    UpstreamReveHead,
    concatenate_all_layers,
    validate_head_variant,
)


@pytest.fixture(autouse=True)
def _preserve_cpu_rng_state():
    with torch.random.fork_rng(devices=[]):
        yield


def _set_deterministic_head_parameters(head: UpstreamReveHead) -> None:
    with torch.no_grad():
        head.norm.weight.copy_(torch.linspace(0.5, 1.5, head.embed_dim))
        head.linear.weight.copy_(torch.arange(head.embed_dim, dtype=torch.float32).view(1, -1))
        head.linear.bias.fill_(0.25)
        if head.query_token is not None:
            head.query_token.copy_(torch.linspace(-0.5, 0.5, head.embed_dim).view(1, 1, -1))


def _expected_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return normalized.to(dtype=x.dtype) * weight


def test_all_variants_return_scalar_and_have_finite_gradients() -> None:
    tokens = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8) / 10
    layers = [tokens, tokens + 1.0, tokens + 2.0]

    for variant in ("last_avg", "last", "all"):
        query = torch.ones(1, 1, 8) if variant != "last_avg" else None
        head = UpstreamReveHead(variant=variant, embed_dim=8, n_outputs=1, dropout=0.0, query_token=query)
        _set_deterministic_head_parameters(head)
        output = head(layers if variant == "all" else tokens)
        assert output.shape == (2, 1)
        assert torch.isfinite(output).all()
        output.sum().backward()
        assert head.linear.weight.grad is not None
        assert torch.isfinite(head.linear.weight.grad).all()
        if variant in {"last", "all"}:
            assert head.query_token.grad is not None
            assert torch.isfinite(head.query_token.grad).all()


def test_last_avg_matches_upstream_mean_norm_dropout_linear_order() -> None:
    tokens = torch.tensor([[[1.0, -2.0, 0.5, 3.0], [0.5, 1.0, -1.0, 2.0]]])
    head = UpstreamReveHead(variant="last_avg", embed_dim=4, n_outputs=1, dropout=0.0)
    _set_deterministic_head_parameters(head)

    mean = tokens.mean(dim=1)
    normalized = _expected_rms_norm(mean, head.norm.weight)
    expected = normalized @ head.linear.weight.T + head.linear.bias

    torch.testing.assert_close(head(tokens), expected)


def test_mean_linear_copy_matches_mean_then_linear() -> None:
    tokens = torch.tensor(
        [
            [[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]],
            [[2.0, 0.0, 1.0], [-1.0, 3.0, 2.0]],
        ]
    )
    head = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(torch.tensor([[0.25, -0.5, 0.75]]))
        head.linear.bias.fill_(0.125)

    expected = F.linear(tokens.mean(dim=1), head.linear.weight, head.linear.bias)

    torch.testing.assert_close(head(tokens), expected)


def test_mean_linear_copy_has_only_linear_parameters() -> None:
    head = MeanLinearCopyHead(embed_dim=4, n_outputs=1)

    assert set(dict(head.named_parameters())) == {"linear.weight", "linear.bias"}
    assert not hasattr(head, "query_token")
    assert not hasattr(head, "norm")
    assert not hasattr(head, "gate_logit")
    assert not any(isinstance(module, nn.Dropout) for module in head.modules())


def test_mean_linear_copy_uses_standalone_linear_initialization() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        head = MeanLinearCopyHead(embed_dim=4, n_outputs=1)
        torch.manual_seed(1234)
        reference = nn.Linear(4, 1)

    torch.testing.assert_close(head.linear.weight, reference.weight)
    torch.testing.assert_close(head.linear.bias, reference.bias)


def test_mean_linear_detached_keeps_mean_branch_out_of_encoder_gradient() -> None:
    tokens = torch.randn(2, 4, 5, requires_grad=True)
    head = MeanLinearDetachedHead(embed_dim=5, n_outputs=1)
    head(tokens).sum().backward()

    assert tokens.grad is None


def test_mean_linear_warmup_starts_as_baseline_and_delays_encoder_gradient() -> None:
    tokens = torch.randn(2, 4, 5, requires_grad=True)
    head = MeanLinearWarmupHead(embed_dim=5, n_outputs=1)
    baseline = MeanLinearCopyHead(embed_dim=5, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    torch.testing.assert_close(head(tokens), baseline(tokens))
    head(tokens).sum().backward()

    assert head.gate.item() == 0.0
    torch.testing.assert_close(tokens.grad, torch.zeros_like(tokens))
    assert head.gate.grad is not None
    assert torch.isfinite(head.gate.grad).all()


def test_mean_linear_gradient_scaled_keeps_forward_and_scales_encoder_gradient() -> None:
    tokens = torch.randn(2, 4, 5, requires_grad=True)
    head = MeanLinearGradientScaledHead(embed_dim=5, n_outputs=1, encoder_gradient_scale=0.1)
    baseline = MeanLinearCopyHead(embed_dim=5, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    torch.testing.assert_close(head(tokens), baseline(tokens))
    head(tokens).sum().backward()

    reference_tokens = tokens.detach().clone().requires_grad_()
    baseline(reference_tokens).sum().backward()
    torch.testing.assert_close(tokens.grad, 0.1 * reference_tokens.grad)


def test_mean_linear_probe_scaled_preserves_forward_and_accelerates_probe_updates() -> None:
    tokens = torch.randn(2, 4, 5, requires_grad=True)
    head = MeanLinearProbeScaledHead(
        embed_dim=5,
        n_outputs=1,
        encoder_gradient_scale=0.1,
        probe_gradient_scale=10.0,
    )
    baseline = MeanLinearCopyHead(embed_dim=5, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    torch.testing.assert_close(head(tokens), baseline(tokens))
    head(tokens).sum().backward()

    reference_tokens = tokens.detach().clone().requires_grad_()
    baseline(reference_tokens).sum().backward()
    torch.testing.assert_close(tokens.grad, 0.1 * reference_tokens.grad)
    torch.testing.assert_close(head.linear.weight.grad, 10.0 * baseline.linear.weight.grad)
    torch.testing.assert_close(head.linear.bias.grad, 10.0 * baseline.linear.bias.grad)


def test_mean_stats_probe_scaled_preserves_forward_and_freezes_encoder_gradient() -> None:
    tokens = torch.randn(2, 4, 5, requires_grad=True)
    head = MeanStatsProbeScaledHead(
        embed_dim=5,
        n_outputs=1,
        encoder_gradient_scale=0.0,
        probe_gradient_scale=10.0,
    )
    baseline = MeanStatsResidualHead(embed_dim=5, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)
        head.correction.weight.copy_(baseline.correction.weight)

    torch.testing.assert_close(head(tokens), baseline(tokens))
    head(tokens).sum().backward()

    assert torch.equal(tokens.grad, torch.zeros_like(tokens))
    assert head.linear.weight.grad is not None
    assert head.correction.weight.grad is not None


def test_mean_rich_stats_gradient_routes_preserve_forward_and_initialization() -> None:
    torch.manual_seed(41)
    control = MeanRichStatsResidualHead(embed_dim=5, n_outputs=1)
    torch.manual_seed(41)
    candidate = MeanRichStatsGradientRoutesHead(
        embed_dim=5,
        n_outputs=1,
        mean_gradient_scale=0.5,
        correction_gradient_scale=0.25,
    )
    tokens = torch.randn(2, 4, 5)

    torch.testing.assert_close(candidate.linear.weight, control.linear.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(candidate.linear.bias, control.linear.bias, rtol=0.0, atol=0.0)
    torch.testing.assert_close(candidate.correction.weight, control.correction.weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(candidate(tokens), control(tokens), rtol=0.0, atol=0.0)
    assert candidate.metadata()["mean_gradient_scale"] == 0.5
    assert candidate.metadata()["correction_gradient_scale"] == 0.25


def test_mean_rich_stats_gradient_routes_scale_only_the_selected_encoder_route() -> None:
    torch.manual_seed(43)
    control = MeanRichStatsResidualHead(embed_dim=4, n_outputs=1)
    torch.manual_seed(43)
    candidate = MeanRichStatsGradientRoutesHead(
        embed_dim=4,
        n_outputs=1,
        mean_gradient_scale=0.5,
        correction_gradient_scale=0.25,
    )
    with torch.no_grad():
        weight = torch.linspace(-0.5, 0.5, candidate.correction.weight.numel()).reshape_as(candidate.correction.weight)
        candidate.correction.weight.copy_(weight)
        control.correction.weight.copy_(weight)

    candidate_tokens = torch.randn(2, 5, 4, requires_grad=True)
    candidate(candidate_tokens).sum().backward()
    candidate_grad = candidate_tokens.grad.detach().clone()

    mean_tokens = candidate_tokens.detach().clone().requires_grad_()
    control.linear(mean_tokens.mean(dim=1)).sum().backward()
    mean_grad = mean_tokens.grad.detach().clone()

    correction_tokens = candidate_tokens.detach().clone().requires_grad_()
    (control.correction_scale * control.correction(control.pool_tokens(correction_tokens))).sum().backward()
    correction_grad = correction_tokens.grad.detach().clone()

    torch.testing.assert_close(
        candidate_grad,
        0.5 * mean_grad + 0.25 * correction_grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_mean_stats_attention_residual_starts_as_mean_linear_with_zero_corrections() -> None:
    tokens = torch.randn(2, 4, 5)
    query = torch.randn(1, 1, 5)
    head = MeanStatsAttentionResidualHead(
        embed_dim=5,
        n_outputs=1,
        query_token=query,
        query_initialization_metadata={"query_initialization": "test"},
    )
    baseline = MeanLinearCopyHead(embed_dim=5, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_mean_attention_gated_starts_as_baseline_and_detaches_correction() -> None:
    tokens = torch.randn(2, 4, 5, requires_grad=True)
    query = torch.randn(1, 1, 5)
    head = MeanAttentionGatedHead(embed_dim=5, n_outputs=1, query_token=query)
    baseline = MeanLinearCopyHead(embed_dim=5, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    torch.testing.assert_close(head(tokens), baseline(tokens))
    assert head.gamma.item() == 0.0
    head(tokens).sum().backward()

    reference_tokens = tokens.detach().clone().requires_grad_()
    baseline(reference_tokens).sum().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)
    assert head.gamma.grad is not None
    assert torch.isfinite(head.gamma.grad).all()


def test_mean_anchor_starts_exactly_as_mean_linear_copy() -> None:
    tokens = torch.tensor(
        [
            [[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]],
            [[2.0, 0.0, 1.0], [-1.0, 3.0, 2.0]],
        ]
    )
    anchor = MeanAnchorHead(embed_dim=3, n_outputs=1)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        anchor.linear.weight.copy_(baseline.linear.weight)
        anchor.linear.bias.copy_(baseline.linear.bias)

    assert torch.equal(anchor.query_token, torch.zeros_like(anchor.query_token))
    assert anchor.gamma.item() == 0.0
    torch.testing.assert_close(anchor.pool_tokens(tokens), tokens.mean(dim=1))
    torch.testing.assert_close(anchor(tokens), baseline(tokens))


def test_mean_anchor_has_simple_trainable_parameters_and_finite_gradients() -> None:
    tokens = torch.randn(2, 4, 5)
    head = MeanAnchorHead(embed_dim=5, n_outputs=1)
    with torch.no_grad():
        head.gamma.fill_(0.25)

    output = head(tokens)
    output.sum().backward()

    assert set(dict(head.named_parameters())) == {
        "query_token",
        "gamma",
        "linear.weight",
        "linear.bias",
    }
    assert torch.isfinite(head.query_token.grad).all()
    assert torch.isfinite(head.gamma.grad).all()
    assert torch.isfinite(head.linear.weight.grad).all()
    assert torch.isfinite(head.linear.bias.grad).all()
    assert not hasattr(head, "norm")
    assert not any(isinstance(module, nn.Dropout) for module in head.modules())


def test_mean_anchor_nonzero_query_allows_gamma_to_learn_from_baseline_start() -> None:
    tokens = torch.tensor([[[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]]])
    head = MeanAnchorHead(
        embed_dim=3,
        n_outputs=1,
        query_token=torch.tensor([[[0.5, -1.0, 2.0]]]),
    )
    output = head(tokens)
    output.sum().backward()

    assert head.gamma.item() == 0.0
    assert torch.isfinite(head.gamma.grad).all()
    assert head.gamma.grad.abs().item() > 0.0


def test_mean_anchor_query_receives_gradient_after_gamma_moves() -> None:
    head = MeanAnchorHead(
        embed_dim=3,
        n_outputs=1,
        query_token=torch.tensor([[[0.5, -1.0, 2.0]]]),
    )
    with torch.no_grad():
        head.gamma.fill_(0.1)

    head(torch.randn(2, 4, 3)).sum().backward()

    assert torch.isfinite(head.query_token.grad).all()
    assert head.query_token.grad.abs().sum().item() > 0.0


def test_mean_anchor_linear_construction_matches_mean_linear_copy_rng() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        anchor = MeanAnchorHead(embed_dim=4, n_outputs=1)
        torch.manual_seed(1234)
        baseline = MeanLinearCopyHead(embed_dim=4, n_outputs=1)

    torch.testing.assert_close(anchor.linear.weight, baseline.linear.weight)
    torch.testing.assert_close(anchor.linear.bias, baseline.linear.bias)


def test_mean_residual_starts_exactly_as_mean_linear_copy() -> None:
    tokens = torch.tensor(
        [[[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]],
         [[2.0, 0.0, 1.0], [-1.0, 3.0, 2.0]]]
    )
    head = MeanResidualHead(
        embed_dim=3,
        n_outputs=1,
        query_token=torch.tensor([[[0.5, -1.0, 2.0]]]),
    )
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    assert torch.equal(head.correction.weight, torch.zeros_like(head.correction.weight))
    mean, residual = head.pool_tokens(tokens)
    torch.testing.assert_close(mean, tokens.mean(dim=1))
    torch.testing.assert_close(head.correction(residual), torch.zeros(mean.shape[0], 1))
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_mean_residual_correction_learns_from_baseline_start() -> None:
    head = MeanResidualHead(
        embed_dim=3,
        n_outputs=1,
        query_token=torch.tensor([[[0.5, -1.0, 2.0]]]),
    )

    head(torch.randn(2, 4, 3)).sum().backward()

    assert set(dict(head.named_parameters())) == {
        "query_token",
        "linear.weight",
        "linear.bias",
        "correction.weight",
    }
    assert torch.isfinite(head.correction.weight.grad).all()
    assert head.correction.weight.grad.abs().sum().item() > 0.0
    assert head.query_token.grad is not None
    assert torch.isfinite(head.query_token.grad).all()


def test_mean_residual_query_learns_after_correction_moves() -> None:
    head = MeanResidualHead(
        embed_dim=3,
        n_outputs=1,
        query_token=torch.tensor([[[0.5, -1.0, 2.0]]]),
    )
    with torch.no_grad():
        head.correction.weight.fill_(0.1)

    head(torch.randn(2, 4, 3)).sum().backward()

    assert head.query_token.grad is not None
    assert torch.isfinite(head.query_token.grad).all()
    assert head.query_token.grad.abs().sum().item() > 0.0


def test_mean_vector_anchor_starts_exactly_as_mean_linear_copy() -> None:
    tokens = torch.randn(2, 4, 3)
    head = MeanVectorAnchorHead(
        embed_dim=3,
        n_outputs=1,
        query_token=torch.tensor([[[0.5, -1.0, 2.0]]]),
    )
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    assert torch.equal(head.gamma, torch.zeros_like(head.gamma))
    torch.testing.assert_close(head.pool_tokens(tokens), tokens.mean(dim=1))
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_mean_vector_anchor_has_featurewise_gate_gradients() -> None:
    head = MeanVectorAnchorHead(
        embed_dim=3,
        n_outputs=1,
        query_token=torch.tensor([[[0.5, -1.0, 2.0]]]),
    )
    head(torch.randn(2, 4, 3)).sum().backward()

    assert set(dict(head.named_parameters())) == {
        "query_token",
        "gamma",
        "linear.weight",
        "linear.bias",
    }
    assert torch.isfinite(head.gamma.grad).all()
    assert head.gamma.grad.abs().sum().item() > 0.0


def test_mean_mlp_residual_starts_exactly_as_mean_linear_copy() -> None:
    tokens = torch.randn(2, 4, 3)
    head = MeanMLPResidualHead(embed_dim=3, n_outputs=1, hidden_dim=4)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    assert torch.equal(head.correction.weight, torch.zeros_like(head.correction.weight))
    torch.testing.assert_close(head.pool_tokens(tokens), tokens.mean(dim=1))
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_mean_mlp_residual_correction_learns_from_baseline_start() -> None:
    head = MeanMLPResidualHead(embed_dim=3, n_outputs=1, hidden_dim=4)
    head(torch.randn(2, 4, 3)).sum().backward()

    assert set(dict(head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "hidden.weight",
        "hidden.bias",
        "correction.weight",
    }
    assert torch.isfinite(head.correction.weight.grad).all()
    assert head.correction.weight.grad.abs().sum().item() > 0.0


def test_mean_stats_residual_starts_exactly_as_mean_linear_copy() -> None:
    tokens = torch.randn(2, 4, 3)
    head = MeanStatsResidualHead(embed_dim=3, n_outputs=1)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    assert torch.equal(head.correction.weight, torch.zeros_like(head.correction.weight))
    stats = head.pool_tokens(tokens)
    assert stats.shape == (2, 6)
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_mean_stats_residual_correction_learns_from_baseline_start() -> None:
    head = MeanStatsResidualHead(embed_dim=3, n_outputs=1)
    head(torch.randn(2, 4, 3)).sum().backward()

    assert set(dict(head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "correction.weight",
    }
    assert torch.isfinite(head.correction.weight.grad).all()
    assert head.correction.weight.grad.abs().sum().item() > 0.0


def test_mean_stats_residual_scales_token_statistics_correction() -> None:
    tokens = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]])
    head = MeanStatsResidualHead(embed_dim=2, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias.zero_()
        head.correction.weight.fill_(1.0)

    raw_correction = head.pool_tokens(tokens).sum(dim=-1, keepdim=True)
    torch.testing.assert_close(head(tokens), 0.5 * raw_correction)


def test_global_stats_residual_starts_exactly_as_mean_linear_copy() -> None:
    torch.manual_seed(17)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    torch.manual_seed(17)
    head = GlobalStatsResidualHead(embed_dim=3, n_outputs=1)
    tokens = torch.randn(2, 4, 3)

    assert torch.equal(head.linear.weight, baseline.linear.weight)
    assert torch.equal(head.linear.bias, baseline.linear.bias)
    assert torch.equal(head.correction.weight, torch.zeros_like(head.correction.weight))
    assert head.pool_tokens(tokens).shape == (2, 4)
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_global_stats_residual_correction_learns_from_baseline_start() -> None:
    head = GlobalStatsResidualHead(embed_dim=3, n_outputs=1)
    head(torch.randn(2, 4, 3)).sum().backward()

    assert set(dict(head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "correction.weight",
    }
    assert torch.isfinite(head.correction.weight.grad).all()
    assert head.correction.weight.grad.abs().sum().item() > 0.0


def test_mean_rich_stats_residual_starts_as_mean_linear_with_four_per_feature_stats() -> None:
    torch.manual_seed(17)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    torch.manual_seed(17)
    head = MeanRichStatsResidualHead(embed_dim=3, n_outputs=1)
    tokens = torch.randn(2, 4, 3)

    assert torch.equal(head.linear.weight, baseline.linear.weight)
    assert torch.equal(head.linear.bias, baseline.linear.bias)
    assert torch.equal(head.correction.weight, torch.zeros_like(head.correction.weight))
    assert head.pool_tokens(tokens).shape == (2, 12)
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_mean_rich_stats_residual_correction_learns_from_baseline_start() -> None:
    head = MeanRichStatsResidualHead(embed_dim=3, n_outputs=1)
    head(torch.randn(2, 4, 3)).sum().backward()

    assert set(dict(head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "correction.weight",
    }
    assert torch.isfinite(head.correction.weight.grad).all()
    assert head.correction.weight.grad.abs().sum().item() > 0.0


def test_multi_query_rich_stats_has_distinct_signed_basis_queries_and_zero_baseline() -> None:
    torch.manual_seed(123)
    head = MultiQueryRichStatsResidualHead(embed_dim=3, n_outputs=1)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    assert head.query_count == 2
    assert head.temperature == 1.0
    torch.testing.assert_close(head.query[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(head.query[1], torch.tensor([-1.0, 0.0, 0.0]))
    assert torch.equal(head.correction.weight, torch.zeros_like(head.correction.weight))
    tokens = torch.randn(2, 5, 3)
    torch.testing.assert_close(head(tokens), baseline(tokens))
    assert head.pool_tokens(tokens).shape == (2, 10 * 3)


def test_multi_query_rich_stats_reports_diversity_and_learns_finite_gradients() -> None:
    head = MultiQueryRichStatsResidualHead(embed_dim=4, n_outputs=1)
    tokens = torch.randn(3, 6, 4, requires_grad=True)
    weights = head.attention_weights(tokens)
    assert weights.shape == (3, 2, 6)
    assert torch.isfinite(weights).all()
    diversity = head.attention_diversity(tokens)
    assert diversity > 1e-4
    metadata = head.metadata()
    assert metadata["query_initialization"] == "signed_basis_pm_e0"
    assert metadata["query_count"] == 2
    head(tokens).sum().backward()
    assert head.query.grad is not None
    assert torch.isfinite(head.query.grad).all()
    assert head.correction.weight.grad is not None


def test_multi_query_rich_stats_rejects_unsupported_screen_settings() -> None:
    with pytest.raises(ValueError, match="query_count"):
        MultiQueryRichStatsResidualHead(embed_dim=3, n_outputs=1, query_count=3)
    with pytest.raises(ValueError, match="temperature"):
        MultiQueryRichStatsResidualHead(embed_dim=3, n_outputs=1, temperature=0.5)


def test_mean_anchor_ensemble_starts_exactly_as_mean_linear_and_learns_gate() -> None:
    torch.manual_seed(17)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    torch.manual_seed(17)
    head = MeanAnchorEnsembleHead(embed_dim=3, n_outputs=1)
    tokens = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]
    )

    assert head.gate_logit.item() == 0.0
    torch.testing.assert_close(head.baseline.linear.weight, baseline.linear.weight)
    torch.testing.assert_close(head.baseline.linear.bias, baseline.linear.bias)
    torch.testing.assert_close(head(tokens), baseline(tokens))

    head(tokens).sum().backward()
    assert head.gate_logit.grad is not None
    assert head.gate_logit.grad.abs().item() > 0.0


def test_mean_anchor_ensemble_gate_is_bounded_and_metadata_is_readable() -> None:
    head = MeanAnchorEnsembleHead(embed_dim=4, n_outputs=1)

    with torch.no_grad():
        head.gate_logit.fill_(100.0)
    assert head.gate_value == pytest.approx(1.0)
    with torch.no_grad():
        head.gate_logit.fill_(-100.0)
    assert head.gate_value == pytest.approx(-1.0)

    metadata = head.metadata()
    assert metadata["gate_parameterization"] == "centered_sigmoid"
    assert metadata["gate_initialization"] == 0.0
    assert metadata["expert"] == "mean_rich_stats_residual"


def test_reliability_shrinkage_starts_as_mean_linear_and_exposes_finite_features() -> None:
    torch.manual_seed(17)
    baseline = MeanLinearCopyHead(embed_dim=4, n_outputs=1)
    torch.manual_seed(17)
    head = MeanReliabilityShrinkageHead(embed_dim=4, n_outputs=1)
    tokens = torch.randn(3, 5, 4)

    torch.testing.assert_close(head.linear.weight, baseline.linear.weight)
    torch.testing.assert_close(head.linear.bias, baseline.linear.bias)
    torch.testing.assert_close(head(tokens), baseline(tokens), rtol=0.0, atol=0.0)

    features = head.reliability_features(tokens)
    gates = head.gate_values(tokens)
    assert features.shape == (3, 3)
    assert gates.shape == (3,)
    assert torch.isfinite(features).all()
    assert torch.isfinite(gates).all()
    assert (gates > 0.0).all()
    assert (gates < head.alpha_max).all()


def test_reliability_shrinkage_correction_and_gate_have_learning_signals() -> None:
    torch.manual_seed(23)
    head = MeanReliabilityShrinkageHead(embed_dim=4, n_outputs=1)
    tokens = torch.randn(3, 6, 4)

    head(tokens).sum().backward()
    assert head.correction.weight.grad is not None
    assert head.correction.weight.grad.abs().sum().item() > 0.0

    head.zero_grad(set_to_none=True)
    with torch.no_grad():
        head.correction.weight.fill_(0.01)
    head(tokens).sum().backward()
    assert head.gate.weight.grad is not None
    assert head.gate.bias.grad is not None
    assert torch.isfinite(head.gate.weight.grad).all()
    assert head.gate.weight.grad.abs().sum().item() > 0.0


def test_reliability_shrinkage_metadata_names_train_only_diagnostics() -> None:
    head = MeanReliabilityShrinkageHead(embed_dim=4, n_outputs=1)
    metadata = head.metadata()

    assert metadata["reliability_features"] == [
        "log1p_dispersion",
        "log1p_mean_token_norm",
        "active_token_fraction",
    ]
    assert metadata["alpha_max"] == 0.5
    assert metadata["gate_initialization"] == -4.0
    assert metadata["correction_initialization"] == "zero"


def test_reliability_stable_reuses_h2_forward_head_and_is_registered() -> None:
    assert reve.validate_head_variant("mean_reliability_stable") == "mean_reliability_stable"
    assert reve.validate_local_head_variant("mean_reliability_stable") == "mean_reliability_stable"

    torch.manual_seed(31)
    h2 = MeanReliabilityShrinkageHead(embed_dim=4, n_outputs=1)
    torch.manual_seed(31)
    stable = reve.MeanReliabilityShrinkageHead(embed_dim=4, n_outputs=1)
    tokens = torch.randn(3, 5, 4)

    assert type(stable) is type(h2)
    assert set(stable.state_dict()) == set(h2.state_dict())
    for key in stable.state_dict():
        torch.testing.assert_close(stable.state_dict()[key], h2.state_dict()[key])
    torch.testing.assert_close(stable(tokens), h2(tokens), rtol=0.0, atol=0.0)


def test_reliability_gate_audit_handles_zero_variance_and_singletons() -> None:
    zero_variance = audit_reliability_gates(
        torch.tensor([0.1, 0.1, 0.1]),
        torch.tensor([1, 1, 2]),
    )
    assert zero_variance["gate_std"] == pytest.approx(0.0)
    assert zero_variance["eta_squared"] == pytest.approx(0.0)
    assert zero_variance["eta_group_count"] == 0
    assert zero_variance["eta_valid"] is False

    grouped = audit_reliability_gates(
        torch.tensor([0.1, 0.1, 0.4, 0.4, 0.9]),
        ["a", "a", "b", "b", "singleton"],
    )
    assert grouped["eta_group_count"] == 2
    assert grouped["eta_sample_count"] == 4
    assert grouped["eta_squared"] == pytest.approx(1.0)


def test_reliability_gate_audit_ignores_missing_subject_ids() -> None:
    audited = audit_reliability_gates(
        torch.tensor([0.1, 0.1, 0.4, 0.4]),
        ["a", None, "b", ""],
    )
    assert audited["eta_group_count"] == 0
    assert audited["eta_squared"] == pytest.approx(0.0)


def test_reliability_gate_audit_marks_nonfinite_values_invalid() -> None:
    audited = audit_reliability_gates(torch.tensor([0.1, float("nan"), 0.2]))
    assert audited["nonfinite_count"] == 1
    assert audited["audit_valid"] is False


def test_grouped_rich_stats_shrinkage_has_zero_gates_and_deterministic_projections() -> None:
    torch.manual_seed(123)
    MeanLinearCopyHead(embed_dim=4, n_outputs=1)
    baseline_rng_after = torch.random.get_rng_state()
    torch.manual_seed(123)
    head = GroupedRichStatsShrinkageHead(embed_dim=4, n_outputs=1)
    grouped_rng_after = torch.random.get_rng_state()

    assert head.statistic_names == ("std", "range", "mad", "mean_abs")
    assert torch.equal(baseline_rng_after, grouped_rng_after)
    assert torch.equal(head.gates, torch.zeros_like(head.gates))
    assert all(torch.isfinite(layer.weight).all() for layer in head.projections)
    assert all(layer.weight.abs().sum().item() > 0.0 for layer in head.projections)
    assert all(torch.equal(layer.bias, torch.zeros_like(layer.bias)) for layer in head.projections)
    for group_index, projection in enumerate(head.projections):
        for row_index, row in enumerate(projection.weight):
            pattern = torch.linspace(-1.0, 1.0, 4)
            offset = (group_index + row_index) % 4
            pattern = torch.roll(pattern, shifts=offset, dims=0)
            if (group_index + row_index) % 2:
                pattern = -pattern
            pattern = pattern / pattern.norm(p=2)
            torch.testing.assert_close(row, pattern)
    assert head.metadata()["projection_initialization"] == (
        "linspace_-1_1_roll_group_plus_row_alternating_sign_l2_normalized_zero_bias"
    )


def test_grouped_rich_stats_shrinkage_is_exact_baseline_at_zero_gates() -> None:
    torch.manual_seed(17)
    baseline = MeanLinearCopyHead(embed_dim=4, n_outputs=1)
    torch.manual_seed(17)
    head = GroupedRichStatsShrinkageHead(embed_dim=4, n_outputs=1)
    tokens = torch.randn(2, 5, 4)

    assert torch.equal(head.linear.weight, baseline.linear.weight)
    assert torch.equal(head.linear.bias, baseline.linear.bias)
    torch.testing.assert_close(head(tokens), baseline(tokens), rtol=0.0, atol=0.0)


def test_grouped_rich_stats_shrinkage_gates_receive_nonzero_gradients() -> None:
    torch.manual_seed(23)
    head = GroupedRichStatsShrinkageHead(embed_dim=4, n_outputs=1)
    tokens = torch.randn(3, 6, 4)

    head(tokens).sum().backward()

    assert head.gates.grad is not None
    assert torch.isfinite(head.gates.grad).all()
    assert (head.gates.grad.abs() > 0.0).all()


def test_grouped_stats_shared_gate_has_one_zero_gate_and_deterministic_projections() -> None:
    torch.manual_seed(123)
    MeanLinearCopyHead(embed_dim=4, n_outputs=1)
    baseline_rng_after = torch.random.get_rng_state()
    torch.manual_seed(123)
    head = GroupedStatsSharedGateHead(embed_dim=4, n_outputs=1)

    assert torch.equal(baseline_rng_after, torch.random.get_rng_state())
    assert head.statistic_names == ("std", "range", "mad", "mean_abs")
    assert head.gate.item() == 0.0
    assert all(torch.isfinite(layer.weight).all() for layer in head.projections)
    assert all(torch.equal(layer.bias, torch.zeros_like(layer.bias)) for layer in head.projections)
    assert head.metadata()["gate_parameterization"] == "shared_scalar"


def test_grouped_stats_shared_gate_is_exact_baseline_at_zero_gate() -> None:
    torch.manual_seed(17)
    baseline = MeanLinearCopyHead(embed_dim=4, n_outputs=1)
    torch.manual_seed(17)
    head = GroupedStatsSharedGateHead(embed_dim=4, n_outputs=1)
    tokens = torch.randn(2, 5, 4)

    assert torch.equal(head.linear.weight, baseline.linear.weight)
    assert torch.equal(head.linear.bias, baseline.linear.bias)
    torch.testing.assert_close(head(tokens), baseline(tokens), rtol=0.0, atol=0.0)


def test_grouped_stats_shared_gate_receives_nonzero_gradient() -> None:
    torch.manual_seed(23)
    head = GroupedStatsSharedGateHead(embed_dim=4, n_outputs=1)
    head(torch.randn(3, 6, 4)).sum().backward()

    assert head.gate.grad is not None
    assert torch.isfinite(head.gate.grad).all()
    assert head.gate.grad.abs().item() > 0.0


def test_temporal_pyramid_stats_residual_starts_as_mean_linear_copy() -> None:
    torch.manual_seed(17)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    torch.manual_seed(17)
    head = TemporalPyramidStatsResidualHead(
        embed_dim=3,
        n_outputs=1,
        segments=2,
        correction_rank=2,
    )
    tokens = torch.tensor(
        [[
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 2.0],
        ]]
    )

    assert torch.equal(head.linear.weight, baseline.linear.weight)
    assert torch.equal(head.linear.bias, baseline.linear.bias)
    assert torch.equal(head.up.weight, torch.zeros_like(head.up.weight))
    assert head.pool_tokens(tokens).shape == (1, 8 * 3)
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_temporal_pyramid_stats_residual_uses_ordered_segments_and_low_rank_gradients() -> None:
    tokens = torch.tensor(
        [[
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 2.0],
        ]],
        requires_grad=True,
    )
    head = TemporalPyramidStatsResidualHead(
        embed_dim=3,
        n_outputs=1,
        segments=2,
        correction_rank=2,
    )

    ordered = head.pool_tokens(tokens)
    permuted = head.pool_tokens(tokens.flip(dims=(1,)))
    assert not torch.equal(ordered, permuted)

    head(tokens).sum().backward()
    assert head.up.weight.grad is not None
    assert torch.isfinite(head.up.weight.grad).all()
    assert head.up.weight.grad.abs().sum().item() > 0.0
    assert head.down.weight.grad is not None
    assert torch.equal(head.down.weight.grad, torch.zeros_like(head.down.weight.grad))


def test_temporal_pyramid_stats_residual_rejects_too_few_tokens() -> None:
    head = TemporalPyramidStatsResidualHead(embed_dim=3, n_outputs=1, segments=2)

    with pytest.raises(AdapterContractError, match="at least as many tokens"):
        head(torch.randn(1, 1, 3))


def test_mean_covariance_residual_starts_as_mean_linear_with_diagonal_covariance() -> None:
    torch.manual_seed(17)
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    torch.manual_seed(17)
    head = MeanCovarianceResidualHead(embed_dim=3, n_outputs=1, projection_rank=4)
    tokens = torch.randn(2, 4, 3)

    assert torch.equal(head.linear.weight, baseline.linear.weight)
    assert torch.equal(head.linear.bias, baseline.linear.bias)
    assert torch.equal(head.up.weight, torch.zeros_like(head.up.weight))
    assert head.pool_tokens(tokens).shape == (2, 3)
    torch.testing.assert_close(head(tokens), baseline(tokens))


def test_mean_covariance_residual_uses_sample_variance_and_learns_low_rank_correction() -> None:
    tokens = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]])
    head = MeanCovarianceResidualHead(embed_dim=2, n_outputs=1, projection_rank=2)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.weight[0, 0] = 1.0
        head.linear.bias.zero_()
        head.down.weight.zero_()
        head.down.weight[0].fill_(1.0)
        head.up.weight.zero_()
        head.up.weight[:, 0].fill_(1.0)

    expected_variance = tokens.var(dim=1, unbiased=True)
    torch.testing.assert_close(head.pool_tokens(tokens), expected_variance)
    torch.testing.assert_close(head(tokens), torch.tensor([[6.0]]))

    head.zero_grad(set_to_none=True)
    head(tokens).sum().backward()
    assert head.up.weight.grad is not None
    assert head.down.weight.grad is not None
    assert torch.isfinite(head.up.weight.grad).all()
    assert torch.isfinite(head.down.weight.grad).all()


def test_mean_covariance_residual_rejects_too_few_tokens() -> None:
    head = MeanCovarianceResidualHead(embed_dim=3, n_outputs=1, projection_rank=4)

    with pytest.raises(AdapterContractError, match="at least two tokens"):
        head(torch.randn(1, 1, 3))


def test_mean_stats_residual_detached_keeps_statistics_out_of_backbone_gradient() -> None:
    tokens = torch.randn(2, 4, 3, requires_grad=True)
    head = MeanStatsResidualDetachedHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        head.correction.weight.fill_(0.1)

    head(tokens).sum().backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    # Only the mean-linear branch may send gradients into the encoder.
    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        baseline.linear.weight.copy_(head.linear.weight)
        baseline.linear.bias.copy_(head.linear.bias)
    reference_tokens = tokens.detach().clone().requires_grad_()
    baseline(reference_tokens).sum().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)


def test_mean_stats_residual_gradient_scaled_scales_only_mean_encoder_gradient() -> None:
    tokens = torch.randn(2, 4, 3, requires_grad=True)
    head = MeanStatsResidualGradientScaledHead(embed_dim=3, n_outputs=1, encoder_gradient_scale=0.1)
    with torch.no_grad():
        head.correction.weight.fill_(0.1)

    head(tokens).sum().backward()

    baseline = MeanLinearCopyHead(embed_dim=3, n_outputs=1)
    with torch.no_grad():
        baseline.linear.weight.copy_(head.linear.weight)
        baseline.linear.bias.copy_(head.linear.bias)
    reference_tokens = tokens.detach().clone().requires_grad_()
    baseline(reference_tokens).sum().backward()
    torch.testing.assert_close(tokens.grad, 0.1 * reference_tokens.grad)


def test_last_matches_upstream_query_attention_formula() -> None:
    tokens = torch.tensor([[[1.0, 0.0, -1.0], [0.5, 2.0, 1.0], [-1.0, 1.0, 0.0]]])
    query = torch.tensor([[[0.5, -1.0, 2.0]]])
    head = UpstreamReveHead(variant="last", embed_dim=3, n_outputs=1, dropout=0.0, query_token=query)
    _set_deterministic_head_parameters(head)

    query = head.query_token.detach()
    scores = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(3)
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights, tokens).squeeze(1)
    expected = _expected_rms_norm(context, head.norm.weight) @ head.linear.weight.T
    expected = expected + head.linear.bias

    torch.testing.assert_close(head(tokens), expected)


def test_all_concatenates_input_then_each_transformer_layer() -> None:
    layers = [
        torch.full((1, 2, 3), 1.0),
        torch.full((1, 2, 3), 2.0),
        torch.full((1, 2, 3), 3.0),
    ]

    combined = concatenate_all_layers(layers)

    assert combined.shape == (1, 6, 3)
    torch.testing.assert_close(combined[:, :2], layers[0])
    torch.testing.assert_close(combined[:, 2:4], layers[1])
    torch.testing.assert_close(combined[:, 4:], layers[2])


def test_mean_layer_linear_excludes_positional_input_and_resolves_final_relative_index() -> None:
    layers = [
        torch.full((1, 2, 3), 100.0),
        torch.full((1, 2, 3), 1.0),
        torch.full((1, 2, 3), 2.0),
        torch.full((1, 2, 3), 3.0),
    ]
    head = MeanLayerLinearHead(embed_dim=3, n_outputs=1, layer_index=-2)

    pooled = head.pool_tokens(layers)

    torch.testing.assert_close(pooled, torch.full((1, 3), 2.0))
    assert head.last_resolved_layer_index == 2
    assert head.metadata()["positional_input_excluded"] is True


@pytest.mark.parametrize(
    "layers",
    [
        torch.randn(2, 3, 4),
        [],
        [torch.randn(2, 3, 4), torch.randn(1, 3, 4)],
        [torch.randn(2, 3, 4), torch.full((2, 3, 4), float("nan"))],
    ],
)
def test_mean_layer_heads_fail_closed_on_invalid_layer_sequences(layers) -> None:
    head = MeanLayerLinearHead(embed_dim=4, n_outputs=1, layer_index=1)

    with pytest.raises(AdapterContractError):
        head.pool_tokens(layers)


def test_mean_layer_mix_starts_exactly_as_final_mean_linear() -> None:
    layers = [
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 4),
    ]
    head = MeanLayerMixHead(embed_dim=4, n_outputs=1)
    baseline = MeanLinearCopyHead(embed_dim=4, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    torch.testing.assert_close(head(layers), baseline(layers[-1]))
    assert head.alpha.item() == 0.0
    assert head.metadata()["layer_indices"] == [1, 2, 3]
    assert head.metadata()["early_layer_indices"] == [1, 2]


def test_mean_layer_mix_fixed_uses_a_frozen_nonzero_alpha() -> None:
    layers = [
        torch.zeros(2, 3, 4),
        torch.ones(2, 3, 4),
        torch.full((2, 3, 4), 3.0),
        torch.full((2, 3, 4), 7.0),
    ]
    head = MeanLayerMixFixedHead(
        embed_dim=4,
        n_outputs=1,
        layer_indices=(-2, -1),
        fixed_alpha=0.5,
    )
    baseline = MeanLinearCopyHead(embed_dim=4, n_outputs=1)
    with torch.no_grad():
        head.linear.weight.copy_(baseline.linear.weight)
        head.linear.bias.copy_(baseline.linear.bias)

    mixed = 7.0 + 0.5 * (3.0 - 7.0)
    expected = baseline.linear(torch.full((2, 4), mixed))
    torch.testing.assert_close(head(layers), expected)
    assert head.alpha.item() == pytest.approx(0.5)
    assert head.alpha.requires_grad is False
    assert head.metadata()["alpha_trainable"] is False
    assert head.metadata()["alpha_initialization"] == "fixed"


def test_mean_layer_mix_rejects_invalid_layer_index() -> None:
    layers = [torch.randn(1, 2, 3), torch.randn(1, 2, 3)]
    head = MeanLayerLinearHead(embed_dim=3, n_outputs=1, layer_index=-2)

    with pytest.raises(AdapterContractError, match="out of range"):
        head.pool_tokens(layers)


def test_invalid_head_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported REVE head variant"):
        validate_head_variant("attention")


@pytest.mark.parametrize(("variant", "expected"), (("mean_linear", "mean_linear"), ("last_avg", "last_avg"), ("last", "last"), ("all", "all"),))
def test_official_head_registry_accepts_only_official_variants(
    variant: str, expected: str
) -> None:
    assert reve.validate_official_head_variant(variant) == expected


def test_last_tuned_is_separate_from_the_official_head_registry() -> None:
    assert validate_head_variant("last_tuned") == "last_tuned"

    with pytest.raises(ValueError, match="official"):
        reve.validate_official_head_variant("last_tuned")

    assert reve.validate_last_tuned_protocol("last_tuned") == "last_tuned"
    for variant in ("mean_linear", "last_avg", "last", "all"):
        with pytest.raises(ValueError):
            reve.validate_last_tuned_protocol(variant)

    torch.manual_seed(123)
    official_last = UpstreamReveHead(variant="last", embed_dim=4, n_outputs=1, dropout=0.0)
    assert official_last.query_initialization == "upstream_random"


def test_last_tuned_uses_a_deterministic_mean_residual_gate() -> None:
    tokens = torch.tensor([[[1.0, 0.0, -1.0], [0.5, 2.0, 1.0], [-1.0, 1.0, 0.0]]])
    head = UpstreamReveHead(variant="last_tuned", embed_dim=3, n_outputs=1, dropout=0.0, query_token=torch.tensor([[[0.5, -1.0, 2.0]]]))

    mean = tokens.mean(dim=1)
    query = head.query_token.detach()
    scores = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(3)
    weights = torch.softmax(scores, dim=-1)
    attention = torch.matmul(weights, tokens).squeeze(1)
    expected = mean + 0.1 * (attention - mean)

    assert head.residual_alpha == pytest.approx(0.1)
    torch.testing.assert_close(head.pool_tokens(tokens), expected)

    pure_attention = torch.matmul(weights, tokens).squeeze(1)
    torch.testing.assert_close(head.pool_tokens(tokens, _test_alpha=1.0), pure_attention)


def test_last_tuned_rejects_finite_tokens_that_overflow_attention() -> None:
    head = UpstreamReveHead(variant="last_tuned", embed_dim=2, n_outputs=1, dropout=0.0, query_token=torch.full((1, 1, 2), 1e38))
    tokens = torch.full((1, 2, 2), 1e38)

    assert torch.isfinite(tokens).all()
    assert torch.isfinite(head.query_token).all()
    with pytest.raises(AdapterContractError, match="scores"):
        head.pool_tokens(tokens)


def test_last_tuned_metadata_is_json_serializable_snapshot() -> None:
    head = UpstreamReveHead(variant="last_tuned", embed_dim=2, n_outputs=1, dropout=0.0, query_token=torch.ones(1, 1, 2))

    metadata = head.tuning_metadata
    assert isinstance(metadata, dict)
    json.dumps(metadata)
    metadata["protocol_class"] = "official"

    assert head.tuning_metadata["protocol_class"] == "tuning"
    assert head.tuning_metadata["query_initialization"] == "provided"


def test_last_tuned_mean_query_is_deterministic_without_rng_leakage() -> None:
    tokens = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    torch.manual_seed(456)
    rng_before = torch.random.get_rng_state()
    mean_query = tokens[:1].mean(dim=1, keepdim=True)
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    torch.manual_seed(789)
    UpstreamReveHead(variant="last", embed_dim=4, n_outputs=1, dropout=0.0, query_token=mean_query)
    control_rng_after = torch.random.get_rng_state()

    torch.manual_seed(789)
    head = UpstreamReveHead(variant="last_tuned", embed_dim=4, n_outputs=1, dropout=0.0, query_token=mean_query)
    tuned_rng_after = torch.random.get_rng_state()

    assert tuple(mean_query.shape) == (1, 1, 4)
    assert torch.isfinite(mean_query).all()
    torch.testing.assert_close(head.query_token.detach(), mean_query)
    torch.testing.assert_close(tuned_rng_after, control_rng_after)


def test_last_tuned_has_finite_query_gate_and_linear_gradients() -> None:
    tokens = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5) / 10
    head = UpstreamReveHead(variant="last_tuned", embed_dim=5, n_outputs=1, dropout=0.0, query_token=torch.ones(1, 1, 5))

    output = head(tokens)
    assert torch.isfinite(output).all()
    output.sum().backward()

    for parameter in (head.query_token, head.gate_logit, head.linear.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("mask_keyword", ("attention_mask", "key_padding_mask", "mask", "padding_mask"))
def test_last_tuned_rejects_every_mask_keyword_even_when_none(
    mask_keyword: str,
) -> None:
    head = UpstreamReveHead(variant="last_tuned", embed_dim=3, n_outputs=1, dropout=0.0, query_token=torch.ones(1, 1, 3))

    with pytest.raises(AdapterContractError, match="mask"):
        head(torch.ones(1, 2, 3), **{mask_keyword: None})


def test_last_rejects_sequence_output_instead_of_falling_back_to_mean() -> None:
    head = UpstreamReveHead(variant="last", embed_dim=3, n_outputs=1, dropout=0.0, query_token=torch.ones(1, 1, 3))

    with pytest.raises(AdapterContractError, match="final token tensor"):
        head([torch.ones(1, 2, 3), torch.ones(1, 2, 3)])


def test_all_rejects_bare_tensor_instead_of_falling_back_to_last() -> None:
    head = UpstreamReveHead(variant="all", embed_dim=3, n_outputs=1, dropout=0.0, query_token=torch.ones(1, 1, 3))

    with pytest.raises(AdapterContractError, match="ordered sequence"):
        head(torch.ones(1, 2, 3))


def test_required_mask_contract_is_rejected() -> None:
    head = UpstreamReveHead(variant="last", embed_dim=3, n_outputs=1, dropout=0.0, query_token=torch.ones(1, 1, 3))

    with pytest.raises(AdapterContractError, match="mask"):
        head(torch.ones(1, 2, 3), attention_mask=torch.ones(1, 2, dtype=torch.bool))


def test_attention_query_matches_explicit_upstream_random_initialization() -> None:
    torch.manual_seed(123)
    expected = torch.randn(1, 1, 8)

    torch.manual_seed(123)
    head = UpstreamReveHead(variant="last", embed_dim=8, n_outputs=1, dropout=0.0)

    torch.testing.assert_close(head.query_token.detach(), expected)
    assert head.query_initialization == "upstream_random"


def test_last_avg_keeps_upstream_unused_query_parameter() -> None:
    head = UpstreamReveHead(variant="last_avg", embed_dim=4, n_outputs=1, dropout=0.0)

    assert head.query_token is not None
    assert head.query_initialization == "upstream_random_unused"
