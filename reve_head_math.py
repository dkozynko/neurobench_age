"""Pure REVE pooling heads and the lazy NeuralBench model adapter."""

from __future__ import annotations

import copy
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .reve_contract import (
        AdapterContractError,
        DEFAULT_UPSTREAM_DROPOUT,
        LAST_TUNED_HEAD_ARCHITECTURE,
        LAST_TUNED_HEAD_SOURCE,
        LAST_TUNED_INITIAL_ALPHA,
        LAST_TUNED_PROTOCOL_CLASS,
        RMS_NORM_EPS,
        UPSTREAM_HEAD_INIT_CUTOFF,
        UPSTREAM_HEAD_INIT_STD,
        validate_local_head_variant,
        validate_last_tuned_protocol,
        validate_upstream_head_variant,
    )
    from .reve_last_tuned import (
        _NEURALBENCH_TRAIN_DUMMY_CONTEXT,
        _encoder_primary_input_key,
        initialize_mean_anchor_query,
        initialize_last_tuned_query,
        _unwrap_reve_module,
    )
except ImportError:
    from reve_contract import (
        AdapterContractError,
        DEFAULT_UPSTREAM_DROPOUT,
        LAST_TUNED_HEAD_ARCHITECTURE,
        LAST_TUNED_HEAD_SOURCE,
        LAST_TUNED_INITIAL_ALPHA,
        LAST_TUNED_PROTOCOL_CLASS,
        RMS_NORM_EPS,
        UPSTREAM_HEAD_INIT_CUTOFF,
        UPSTREAM_HEAD_INIT_STD,
        validate_local_head_variant,
        validate_last_tuned_protocol,
        validate_upstream_head_variant,
    )
    from reve_last_tuned import (
        _NEURALBENCH_TRAIN_DUMMY_CONTEXT,
        _encoder_primary_input_key,
        initialize_mean_anchor_query,
        initialize_last_tuned_query,
        _unwrap_reve_module,
    )

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
        raise AdapterContractError(f"the fixed-window REVE adapter does not accept a required mask; received {supplied}")


def _validate_final_tokens(tokens: Any, *, embed_dim: int) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise AdapterContractError("last/last_avg require a final token tensor with shape [batch, tokens, dim]")
    if tokens.shape[-1] != embed_dim:
        raise AdapterContractError(f"final token tensor has embedding dimension {tokens.shape[-1]}, expected {embed_dim}")
    if tokens.shape[1] <= 0:
        raise AdapterContractError("final token tensor must contain at least one token")
    return tokens



def _validate_query_shape(query_token: Any, *, embed_dim: int) -> torch.Tensor:
    """Validate the shared ``[1, 1, embed_dim]`` query-token shape."""

    if not isinstance(query_token, torch.Tensor) or tuple(query_token.shape) != (
        1,
        1,
        embed_dim,
    ):
        shape = getattr(query_token, "shape", None)
        raise AdapterContractError(f"cls_query_token must have shape [1, 1, {embed_dim}], got {shape}")
    return query_token


def _validate_query_token(query_token: Any, *, embed_dim: int) -> torch.Tensor:
    query_token = _validate_query_shape(query_token, embed_dim=embed_dim)
    if not query_token.is_floating_point():
        raise AdapterContractError("cls_query_token must have a floating-point dtype")
    if not torch.isfinite(query_token).all():
        raise AdapterContractError("cls_query_token must contain only finite values")
    return query_token


def concatenate_all_layers(
    layers: Sequence[torch.Tensor], *, embed_dim: int | None = None
) -> torch.Tensor:
    """Concatenate REVE's positional input and transformer outputs by tokens."""

    if isinstance(layers, torch.Tensor) or not isinstance(layers, (list, tuple)):
        raise AdapterContractError("all requires an ordered sequence of [batch, tokens, dim] tensors")
    if not layers:
        raise AdapterContractError("all requires at least one encoder output sequence")

    first = layers[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 3:
        raise AdapterContractError("all requires every encoder output to have shape [batch, tokens, dim]")
    expected_shape = (first.shape[0], first.shape[-1])
    if embed_dim is not None and first.shape[-1] != embed_dim:
        raise AdapterContractError(f"all output has embedding dimension {first.shape[-1]}, expected {embed_dim}")
    for index, layer in enumerate(layers):
        if not isinstance(layer, torch.Tensor) or layer.ndim != 3:
            raise AdapterContractError(f"all requires every encoder output to have shape [batch, tokens, dim]; item {index} is invalid")
        if (layer.shape[0], layer.shape[-1]) != expected_shape:
            raise AdapterContractError(
                "all encoder outputs must share batch and embedding dimensions; "
                f"item {index} has shape {tuple(layer.shape)}"
            )
        if layer.shape[1] <= 0:
            raise AdapterContractError(f"all encoder output {index} has no tokens")
    return torch.cat(tuple(layers), dim=1)


class MeanLinearCopyHead(nn.Module):
    """Readable local copy of NeuralBench's official mean-plus-linear head."""

    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, n_outputs)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Average the final REVE token sequence over its token dimension."""

        return _validate_final_tokens(tokens, embed_dim=self.embed_dim).mean(dim=1)

    def forward(self, tokens: Any) -> torch.Tensor:
        return self.linear(self.pool_tokens(tokens))


class MeanLinearDetachedHead(MeanLinearCopyHead):
    """Mean-linear probe that keeps the pretrained encoder fixed."""

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Average tokens without sending gradients back into the encoder."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        return tokens.detach().mean(dim=1)


class MeanLinearWarmupHead(nn.Module):
    """Mean-linear baseline with a gated residual fine-tuning path."""

    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        # Build the baseline first so it has the same RNG initialization as
        # MeanLinearCopyHead.  The residual starts behind a zero gate.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.residual = nn.Linear(embed_dim, n_outputs)
        self.gate = nn.Parameter(torch.zeros(()))

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Average final tokens for both the baseline and residual paths."""

        return _validate_final_tokens(tokens, embed_dim=self.embed_dim).mean(dim=1)

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        baseline = self.linear(mean.detach())
        return baseline + self.gate * self.residual(mean)


class MeanLinearGradientScaledHead(MeanLinearCopyHead):
    """Mean-linear probe with a controlled encoder-gradient scale."""

    def __init__(self, *, embed_dim: int, n_outputs: int, encoder_gradient_scale: float = 0.1):
        super().__init__(embed_dim=embed_dim, n_outputs=n_outputs)
        if not 0.0 <= encoder_gradient_scale <= 1.0:
            raise ValueError("encoder_gradient_scale must be in [0, 1]")
        self.encoder_gradient_scale = float(encoder_gradient_scale)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Keep mean forward values while scaling only encoder gradients."""

        mean = _validate_final_tokens(tokens, embed_dim=self.embed_dim).mean(dim=1)
        return mean.detach() + self.encoder_gradient_scale * (mean - mean.detach())


class MeanLinearProbeScaledHead(MeanLinearCopyHead):
    """Mean-linear probe with separate encoder and probe gradient scales."""

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        encoder_gradient_scale: float = 0.1,
        probe_gradient_scale: float = 10.0,
    ):
        super().__init__(embed_dim=embed_dim, n_outputs=n_outputs)
        if not 0.0 <= encoder_gradient_scale <= 1.0:
            raise ValueError("encoder_gradient_scale must be in [0, 1]")
        if probe_gradient_scale <= 0.0:
            raise ValueError("probe_gradient_scale must be positive")
        self.encoder_gradient_scale = float(encoder_gradient_scale)
        self.probe_gradient_scale = float(probe_gradient_scale)
    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Keep mean forward values while scaling encoder gradients."""

        mean = _validate_final_tokens(tokens, embed_dim=self.embed_dim).mean(dim=1)
        return mean.detach() + self.encoder_gradient_scale * (mean - mean.detach())

    def forward(self, tokens: Any) -> torch.Tensor:
        mean = self.pool_tokens(tokens)
        # Detach-based interpolation preserves the parameter's forward value
        # while multiplying its backward gradient by the requested scale.
        weight = self.linear.weight.detach() + self.probe_gradient_scale * (
            self.linear.weight - self.linear.weight.detach()
        )
        bias = self.linear.bias.detach() + self.probe_gradient_scale * (
            self.linear.bias - self.linear.bias.detach()
        )
        return F.linear(
            mean,
            weight,
            bias,
        )


class MeanAnchorHead(nn.Module):
    """Mean-linear baseline with a learnable, zero-gamma attention residual."""

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        query_token: torch.Tensor | None = None,
        query_initialization_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        if query_token is None:
            query_token = torch.zeros(1, 1, embed_dim)
            self.query_initialization = "zero_uniform_attention"
        else:
            query_token = _validate_query_token(query_token, embed_dim=embed_dim)
            self.query_initialization = "provided"
        if query_initialization_metadata is not None:
            initialization_name = query_initialization_metadata.get("query_initialization")
            if not isinstance(initialization_name, str):
                raise AdapterContractError("mean_anchor query initialization metadata must name its source")
            self.query_initialization = initialization_name
        self.gamma_initialization = "zero"
        self.query_token = nn.Parameter(query_token.detach().clone())
        self.gamma = nn.Parameter(torch.zeros(()))
        # Keep this construction after the zero-only parameters so the Linear
        # consumes the same RNG position as MeanLinearCopyHead.
        self.linear = nn.Linear(embed_dim, n_outputs)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Pool final tokens, starting as exact mean pooling."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        query = self.query_token.expand(tokens.shape[0], -1, -1)
        scores = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, tokens).squeeze(1)
        return mean + self.gamma * (attention - mean)

    def forward(self, tokens: Any) -> torch.Tensor:
        return self.linear(self.pool_tokens(tokens))


class MeanResidualHead(nn.Module):
    """Mean-linear baseline plus a zero-initialized attention correction."""

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        query_token: torch.Tensor | None = None,
        query_initialization_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        if query_token is None:
            query_token = torch.zeros(1, 1, embed_dim)
            self.query_initialization = "zero_uniform_attention"
        else:
            query_token = _validate_query_token(query_token, embed_dim=embed_dim)
            self.query_initialization = "provided"
        if query_initialization_metadata is not None:
            initialization_name = query_initialization_metadata.get("query_initialization")
            if not isinstance(initialization_name, str):
                raise AdapterContractError("mean_residual query initialization metadata must name its source")
            self.query_initialization = initialization_name

        self.query_token = nn.Parameter(query_token.detach().clone())
        # Construct the baseline linear first, so it has exactly the same RNG
        # position as MeanLinearCopyHead. The correction starts at zero, which
        # makes the complete head exactly equal to mean_linear at initialization.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.correction = nn.Linear(embed_dim, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)

    def pool_tokens(self, tokens: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean features and the query-attention residual features."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        query = self.query_token.expand(tokens.shape[0], -1, -1)
        scores = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, tokens).squeeze(1)
        return mean, attention - mean

    def forward(self, tokens: Any) -> torch.Tensor:
        mean, residual = self.pool_tokens(tokens)
        return self.linear(mean) + self.correction(residual)


class MeanVectorAnchorHead(nn.Module):
    """Mean-linear baseline with a featurewise zero-gamma attention residual."""

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        query_token: torch.Tensor | None = None,
        query_initialization_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        if query_token is None:
            query_token = torch.zeros(1, 1, embed_dim)
            self.query_initialization = "zero_uniform_attention"
        else:
            query_token = _validate_query_token(query_token, embed_dim=embed_dim)
            self.query_initialization = "provided"
        if query_initialization_metadata is not None:
            initialization_name = query_initialization_metadata.get("query_initialization")
            if not isinstance(initialization_name, str):
                raise AdapterContractError("mean_vector_anchor query initialization metadata must name its source")
            self.query_initialization = initialization_name

        self.query_token = nn.Parameter(query_token.detach().clone())
        self.gamma = nn.Parameter(torch.zeros(embed_dim))
        self.linear = nn.Linear(embed_dim, n_outputs)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Pool final tokens with one learnable residual gate per feature."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        query = self.query_token.expand(tokens.shape[0], -1, -1)
        scores = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, tokens).squeeze(1)
        return mean + self.gamma * (attention - mean)

    def forward(self, tokens: Any) -> torch.Tensor:
        return self.linear(self.pool_tokens(tokens))


class MeanMLPResidualHead(nn.Module):
    """Mean-linear baseline plus a zero-initialized nonlinear correction."""

    def __init__(self, *, embed_dim: int, n_outputs: int, hidden_dim: int | None = None):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        if hidden_dim is None:
            hidden_dim = max(32, min(256, embed_dim // 2))
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Construct the baseline probe first; its RNG sequence is identical to
        # MeanLinearCopyHead. The correction starts at exactly zero.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.hidden = nn.Linear(embed_dim, hidden_dim)
        self.correction = nn.Linear(hidden_dim, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return the mean representation used by both baseline and correction."""

        return _validate_final_tokens(tokens, embed_dim=self.embed_dim).mean(dim=1)

    def forward(self, tokens: Any) -> torch.Tensor:
        mean = self.pool_tokens(tokens)
        nonlinear_features = torch.nn.functional.gelu(self.hidden(mean))
        return self.linear(mean) + self.correction(nonlinear_features)


class MeanStatsResidualHead(nn.Module):
    """Mean-linear baseline plus a conservative token-statistics correction."""

    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        self.correction_scale = 0.5
        # Keep the baseline probe first so it follows the same construction
        # and initialization path as MeanLinearCopyHead.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.correction = nn.Linear(2 * embed_dim, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return per-feature standard deviation and range for final tokens."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        standard_deviation = tokens.std(dim=1, unbiased=False)
        value_range = tokens.amax(dim=1) - tokens.amin(dim=1)
        return torch.cat((standard_deviation, value_range), dim=-1)

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        return self.linear(mean) + self.correction_scale * self.correction(self.pool_tokens(tokens))


class GlobalStatsResidualHead(nn.Module):
    """Mean-linear baseline plus a small correction from global token statistics."""

    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        self.correction_scale = 0.5
        # Build the baseline first so its initialization is identical to the
        # mean-linear control. The zero correction gives exact initial parity.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.correction = nn.Linear(4, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Summarize each record with four scalar token-distribution statistics."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        standard_deviation = tokens.std(dim=1, unbiased=False).mean(dim=1, keepdim=True)
        value_range = (tokens.amax(dim=1) - tokens.amin(dim=1)).mean(dim=1, keepdim=True)
        mean_absolute_deviation = (
            (tokens - mean.unsqueeze(1)).abs().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        )
        mean_absolute_value = tokens.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return torch.cat(
            (standard_deviation, value_range, mean_absolute_deviation, mean_absolute_value),
            dim=1,
        )

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        return self.linear(mean) + self.correction_scale * self.correction(self.pool_tokens(tokens))


class MeanRichStatsResidualHead(nn.Module):
    """Mean-linear baseline plus a zero correction from four per-feature stats."""

    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        self.correction_scale = 0.5
        # Construct the baseline first so its initialization is identical to
        # MeanLinearCopyHead.  The correction starts at zero, preserving exact
        # mean_linear predictions at initialization.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.correction = nn.Linear(4 * embed_dim, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return per-feature spread and magnitude statistics for final tokens."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1, keepdim=True)
        standard_deviation = tokens.std(dim=1, unbiased=False)
        value_range = tokens.amax(dim=1) - tokens.amin(dim=1)
        mean_absolute_deviation = (tokens - mean).abs().mean(dim=1)
        mean_absolute_value = tokens.abs().mean(dim=1)
        return torch.cat(
            (standard_deviation, value_range, mean_absolute_deviation, mean_absolute_value),
            dim=-1,
        )

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        return self.linear(mean) + self.correction_scale * self.correction(self.pool_tokens(tokens))


class MultiQueryRichStatsResidualHead(nn.Module):
    """Mean-linear probe with two deterministic, diverse attention queries.

    The zero correction keeps the forward output exactly equal to mean-linear
    at initialization.  The queries only determine the additional statistics,
    so this variant changes one interpretable feature extractor at a time.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        query_count: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        if query_count != 2:
            raise ValueError("query_count is fixed at 2 for the strict screen")
        if not math.isfinite(float(temperature)) or float(temperature) != 1.0:
            raise ValueError("temperature is fixed at 1.0 for the strict screen")
        self.embed_dim = embed_dim
        self.query_count = int(query_count)
        self.temperature = float(temperature)
        self.correction_scale = 0.5
        # Deterministic signed-basis queries avoid both random initialization
        # and a symmetry in which the two attention distributions collapse.
        query = torch.zeros(self.query_count, embed_dim)
        query[0, 0] = 1.0
        query[1, 0] = -1.0
        self.query = nn.Parameter(query)
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.correction = nn.Linear(10 * embed_dim, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)

    def attention_weights(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        scores = torch.einsum("kd,btd->bkt", self.query, tokens)
        scores = scores / math.sqrt(self.embed_dim) / self.temperature
        return torch.softmax(scores, dim=-1)

    def _query_statistics(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_mean = torch.einsum("bkt,btd->bkd", weights, tokens)
        centered = tokens.unsqueeze(1) - weighted_mean.unsqueeze(2)
        weighted_std = torch.sqrt(
            torch.einsum("bkt,bktd->bkd", weights, centered.square()).clamp_min(0.0)
            + 1e-8
        )
        value_range = (
            tokens.amax(dim=1).unsqueeze(1) - tokens.amin(dim=1).unsqueeze(1)
        ).expand(-1, self.query_count, -1)
        weighted_mad = torch.einsum("bkt,bktd->bkd", weights, centered.abs())
        weighted_abs = torch.einsum("bkt,btd->bkd", weights, tokens.abs())
        # Five D-dimensional statistics per query: location, spread, range,
        # deviation, and magnitude.
        return torch.cat(
            (weighted_mean, weighted_std, value_range, weighted_mad, weighted_abs),
            dim=-1,
        )

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        return self._query_statistics(tokens, self.attention_weights(tokens)).flatten(1)

    def attention_diversity(self, tokens: Any) -> float:
        weights = self.attention_weights(tokens)
        if weights.shape[1] != 2:
            raise RuntimeError("multi-query diversity requires exactly two queries")
        return float((weights[:, 0] - weights[:, 1]).abs().sum(dim=-1).mean().detach().cpu())

    def metadata(self) -> dict[str, Any]:
        return {
            "matched_baseline": "mean_linear",
            "query_count": self.query_count,
            "temperature": self.temperature,
            "query_initialization": "signed_basis_pm_e0",
            "correction_initialization": "zero",
            "correction_scale": self.correction_scale,
            "statistic_count_per_query": 5,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        return self.linear(tokens.mean(dim=1)) + self.correction_scale * self.correction(
            self.pool_tokens(tokens)
        )


class MeanRichStatsGradientRoutesHead(MeanRichStatsResidualHead):
    """Rich-statistics head with independently scaled encoder gradients."""

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        mean_gradient_scale: float = 0.5,
        correction_gradient_scale: float = 1.0,
    ):
        super().__init__(embed_dim=embed_dim, n_outputs=n_outputs)
        for name, value in (
            ("mean_gradient_scale", mean_gradient_scale),
            ("correction_gradient_scale", correction_gradient_scale),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 2.0:
                raise ValueError(f"{name} must be finite and in [0, 2]")
        self.mean_gradient_scale = float(mean_gradient_scale)
        self.correction_gradient_scale = float(correction_gradient_scale)

    @staticmethod
    def _scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
        """Keep a tensor's value while multiplying its backward derivative."""

        return value.detach() + float(scale) * (value - value.detach())

    def metadata(self) -> dict[str, Any]:
        """Return stable metadata for the strict route audit."""

        return {
            "matched_baseline": "mean_rich_stats_residual",
            "gradient_target": "encoder_routes_only",
            "mean_gradient_scale": self.mean_gradient_scale,
            "correction_gradient_scale": self.correction_gradient_scale,
            "correction_scale": self.correction_scale,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        statistics = self.pool_tokens(tokens)
        mean_route = self._scale_gradient(mean, self.mean_gradient_scale)
        correction_route = self._scale_gradient(statistics, self.correction_gradient_scale)
        # Expose route tensors only for H7's optional training diagnostic.
        # This does not change the forward value or parameterization.
        if torch.is_grad_enabled():
            for value in (mean, statistics, mean_route, correction_route):
                if value.requires_grad:
                    value.retain_grad()
            self._last_gradient_route_tensors = {
                "mean": mean,
                "statistics": statistics,
                "mean_route": mean_route,
                "correction_route": correction_route,
            }
        return self.linear(mean_route) + self.correction_scale * self.correction(correction_route)


class MeanAnchorEnsembleHead(nn.Module):
    """Blend mean-linear predictions with a rich-statistics expert.

    The baseline branch is constructed first and is therefore an exact local
    ``mean_linear`` copy at initialization.  A centered sigmoid gate starts at
    zero, so the initial prediction is exactly the baseline while the gate can
    learn whether the independently initialized expert is useful.
    """

    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        self.n_outputs = n_outputs
        self.baseline = MeanLinearCopyHead(embed_dim=embed_dim, n_outputs=n_outputs)
        self.expert = MeanRichStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    @property
    def gate_value(self) -> float:
        """Return the bounded scalar interpolation weight."""

        return float((2.0 * torch.sigmoid(self.gate_logit).detach() - 1.0).item())

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return the anchored mean representation for embedding consumers."""

        return self.baseline.pool_tokens(tokens)

    def metadata(self) -> dict[str, Any]:
        """Return stable, JSON-serializable architecture metadata."""

        return {
            "expert": "mean_rich_stats_residual",
            "gate_parameterization": "centered_sigmoid",
            "gate_initialization": 0.0,
            "gate_range": [-1.0, 1.0],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, tokens: Any) -> torch.Tensor:
        baseline_prediction = self.baseline(tokens)
        expert_prediction = self.expert(tokens)
        gate = 2.0 * torch.sigmoid(self.gate_logit) - 1.0
        return baseline_prediction + gate * (expert_prediction - baseline_prediction)


class MeanReliabilityShrinkageHead(nn.Module):
    """Mean-linear probe with a train-only, input-conditioned shrinkage gate.

    The correction is built from rich per-feature token statistics.  A small
    gate reads only dispersion, token norm, and finite-token fraction, so each
    recording can receive a different correction strength without using age or
    split-specific information.
    """

    reliability_feature_names = (
        "log1p_dispersion",
        "log1p_mean_token_norm",
        "active_token_fraction",
    )

    def __init__(self, *, embed_dim: int, n_outputs: int, alpha_max: float = 0.5):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        if not 0.0 < alpha_max <= 1.0:
            raise ValueError("alpha_max must be in (0, 1]")
        self.embed_dim = embed_dim
        self.n_outputs = n_outputs
        self.alpha_max = float(alpha_max)
        self.correction_scale = 1.0
        # Construct the baseline first so its initialization is identical to
        # MeanLinearCopyHead.  The correction starts at zero, preserving exact
        # mean_linear predictions while the gate is already well-defined.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.correction = nn.Linear(4 * embed_dim, n_outputs, bias=False)
        nn.init.zeros_(self.correction.weight)
        self.gate = nn.Linear(len(self.reliability_feature_names), 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        # A detached CPU snapshot lets the strict validation callback audit
        # the gate without retaining a computation graph or touching test data.
        self._last_gate_values: torch.Tensor | None = None

    def _safe_tokens(self, tokens: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        safe = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        active = torch.isfinite(tokens).all(dim=-1) & (safe.norm(dim=-1) > 1e-6)
        return safe, active

    def reliability_features(self, tokens: Any) -> torch.Tensor:
        """Return finite diagnostics available from the current recording only."""

        safe, valid = self._safe_tokens(tokens)
        mean = safe.mean(dim=1, keepdim=True)
        dispersion = (safe - mean).abs().mean(dim=(1, 2))
        mean_token_norm = safe.norm(dim=-1).mean(dim=1)
        active_token_fraction = valid.to(dtype=safe.dtype).mean(dim=1)
        return torch.stack(
            (
                torch.log1p(dispersion),
                torch.log1p(mean_token_norm),
                active_token_fraction,
            ),
            dim=-1,
        )

    def gate_values(self, tokens: Any) -> torch.Tensor:
        """Return one bounded correction strength per recording."""

        features = self.reliability_features(tokens)
        values = self.alpha_max * torch.sigmoid(self.gate(features)).squeeze(-1)
        self._last_gate_values = values.detach().cpu()
        return values

    def rich_statistics(self, tokens: Any) -> torch.Tensor:
        """Return four per-feature statistics used by the correction branch."""

        safe, _ = self._safe_tokens(tokens)
        mean = safe.mean(dim=1, keepdim=True)
        standard_deviation = safe.std(dim=1, unbiased=False)
        value_range = safe.amax(dim=1) - safe.amin(dim=1)
        mean_absolute_deviation = (safe - mean).abs().mean(dim=1)
        mean_absolute_value = safe.abs().mean(dim=1)
        return torch.cat(
            (standard_deviation, value_range, mean_absolute_deviation, mean_absolute_value),
            dim=-1,
        )

    def metadata(self) -> dict[str, Any]:
        """Return stable, JSON-serializable architecture metadata."""

        return {
            "reliability_features": list(self.reliability_feature_names),
            "alpha_max": self.alpha_max,
            "gate_parameterization": "input_conditioned_sigmoid",
            "gate_initialization": -4.0,
            "correction_initialization": "zero",
            "correction_statistics": ["std", "range", "mad", "mean_abs"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return the mean representation used by the baseline branch."""

        return _validate_final_tokens(tokens, embed_dim=self.embed_dim).mean(dim=1)

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        correction = self.correction(self.rich_statistics(tokens))
        alpha = self.gate_values(tokens).unsqueeze(-1)
        return self.linear(mean) + self.correction_scale * alpha * correction


def audit_reliability_gates(
    gate_values: Any,
    subject_ids: Any = None,
    *,
    alpha_max: float = 0.5,
    saturation_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Summarize gate health and an optional training-only identity proxy.

    ``subject_ids`` are deliberately optional: validation loaders may not
    expose identifiers.  When present, singleton and missing groups are
    excluded from the between-subject variance calculation; no labels are
    consumed.  The result is JSON-serializable and safe for epoch artifacts.
    """

    if not 0.0 < float(alpha_max) <= 1.0:
        raise ValueError("alpha_max must be in (0, 1]")
    if float(saturation_epsilon) < 0.0:
        raise ValueError("saturation_epsilon must be non-negative")
    values = torch.as_tensor(gate_values, dtype=torch.float64).reshape(-1)
    finite_mask = torch.isfinite(values)
    nonfinite_count = int((~finite_mask).sum().item())
    finite_values = values[finite_mask]
    if finite_values.numel() == 0:
        raise ValueError("gate_values must contain at least one finite value")
    low_cutoff = float(saturation_epsilon)
    high_cutoff = float(alpha_max) - low_cutoff
    summary: dict[str, Any] = {
        "sample_count": int(finite_values.numel()),
        "raw_sample_count": int(values.numel()),
        "nonfinite_count": nonfinite_count,
        "audit_valid": nonfinite_count == 0,
        "gate_mean": float(finite_values.mean().item()),
        "gate_std": float(finite_values.std(unbiased=False).item()),
        "low_saturation_fraction": float((finite_values <= low_cutoff).double().mean().item()),
        "high_saturation_fraction": float((finite_values >= high_cutoff).double().mean().item()),
        "combined_saturation_fraction": float(
            ((finite_values <= low_cutoff) | (finite_values >= high_cutoff)).double().mean().item()
        ),
        "alpha_max": float(alpha_max),
        "saturation_epsilon": float(saturation_epsilon),
        "eta_squared": 0.0,
        "eta_group_count": 0,
        "eta_sample_count": 0,
        "eta_valid": False,
        "eta_reason": "subject_ids_unavailable",
    }
    if subject_ids is None:
        return summary

    if isinstance(subject_ids, torch.Tensor):
        raw_ids = subject_ids.reshape(-1).tolist()
    else:
        raw_ids = list(subject_ids)
    if len(raw_ids) != len(values):
        raise ValueError("subject_ids must have the same number of entries as gate_values")

    def _normalise_id(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.item()
        if value is None:
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    groups: dict[Any, list[float]] = {}
    for value, raw_id in zip(values.tolist(), raw_ids):
        if not math.isfinite(float(value)):
            continue
        key = _normalise_id(raw_id)
        if key is None:
            continue
        groups.setdefault(key, []).append(float(value))
    usable = [group for group in groups.values() if len(group) >= 2]
    if len(usable) < 2 or sum(len(group) for group in usable) < 4:
        summary["eta_reason"] = "insufficient_repeated_subject_groups"
        return summary
    pooled = [value for group in usable for value in group]
    pooled_mean = sum(pooled) / len(pooled)
    total_ss = sum((value - pooled_mean) ** 2 for value in pooled)
    if total_ss <= 1e-12:
        eta_squared = 0.0
    else:
        between_ss = sum(len(group) * ((sum(group) / len(group)) - pooled_mean) ** 2 for group in usable)
        eta_squared = max(0.0, min(1.0, between_ss / total_ss))
    summary.update(
        {
            "eta_squared": float(eta_squared),
            "eta_group_count": len(usable),
            "eta_sample_count": len(pooled),
            "eta_valid": True,
            "eta_reason": None,
        }
    )
    return summary


def _build_deterministic_statistic_projections(
    *, embed_dim: int, count: int
) -> nn.ModuleList:
    """Build reproducible D-to-D projections without changing the RNG stream."""

    rng_state = torch.random.get_rng_state()
    try:
        projections = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=True) for _ in range(count)]
        )
    finally:
        torch.random.set_rng_state(rng_state)
    for group_index, projection in enumerate(projections):
        rows = []
        for row_index in range(embed_dim):
            pattern = torch.linspace(-1.0, 1.0, embed_dim)
            if embed_dim == 1:
                pattern = torch.ones(1)
            else:
                offset = (group_index + row_index) % embed_dim
                pattern = torch.roll(pattern, shifts=offset, dims=0)
                if (group_index + row_index) % 2:
                    pattern = -pattern
            rows.append(pattern / pattern.norm(p=2))
        with torch.no_grad():
            projection.weight.copy_(torch.stack(rows, dim=0))
            projection.bias.zero_()
    return projections


class GroupedRichStatsShrinkageHead(nn.Module):
    """Mean-linear baseline with one zero-start gate per rich-stat group."""

    statistic_names = ("std", "range", "mad", "mean_abs")
    projection_initialization = (
        "linspace_-1_1_roll_group_plus_row_alternating_sign_l2_normalized_zero_bias"
    )

    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        self.n_outputs = n_outputs
        self.correction_scale = 0.5
        # Construct the baseline first to preserve MeanLinearCopyHead's RNG
        # order and exact baseline parameters at initialization.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.projections = _build_deterministic_statistic_projections(
            embed_dim=embed_dim, count=len(self.statistic_names)
        )
        self.gates = nn.Parameter(torch.zeros(len(self.statistic_names)))

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return four per-feature statistics concatenated by group."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1, keepdim=True)
        standard_deviation = tokens.std(dim=1, unbiased=False)
        value_range = tokens.amax(dim=1) - tokens.amin(dim=1)
        mean_absolute_deviation = (tokens - mean).abs().mean(dim=1)
        mean_absolute_value = tokens.abs().mean(dim=1)
        return torch.cat(
            (standard_deviation, value_range, mean_absolute_deviation, mean_absolute_value),
            dim=-1,
        )

    def metadata(self) -> dict[str, Any]:
        """Return stable, JSON-serializable architecture metadata."""

        return {
            "statistic_groups": list(self.statistic_names),
            "correction_scale": self.correction_scale,
            "gate_parameterization": "direct_scalar",
            "gate_initialization": 0.0,
            "projection_initialization": self.projection_initialization,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        grouped_statistics = self.pool_tokens(tokens).chunk(len(self.statistic_names), dim=-1)
        correction = sum(
            gate * projection(statistic)
            for gate, projection, statistic in zip(self.gates, self.projections, grouped_statistics)
        )
        return self.linear(mean + self.correction_scale * correction)


class GroupedStatsSharedGateHead(nn.Module):
    """Mean-linear baseline with one shared gate for rich-stat corrections."""

    statistic_names = GroupedRichStatsShrinkageHead.statistic_names
    projection_initialization = GroupedRichStatsShrinkageHead.projection_initialization

    def __init__(self, *, embed_dim: int, n_outputs: int):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        self.n_outputs = n_outputs
        self.correction_scale = 0.5
        # Build the baseline first so zero gate gives exact mean-linear parity.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.projections = _build_deterministic_statistic_projections(
            embed_dim=embed_dim, count=len(self.statistic_names)
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return four per-feature statistics concatenated by group."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1, keepdim=True)
        standard_deviation = tokens.std(dim=1, unbiased=False)
        value_range = tokens.amax(dim=1) - tokens.amin(dim=1)
        mean_absolute_deviation = (tokens - mean).abs().mean(dim=1)
        mean_absolute_value = tokens.abs().mean(dim=1)
        return torch.cat(
            (standard_deviation, value_range, mean_absolute_deviation, mean_absolute_value),
            dim=-1,
        )

    def metadata(self) -> dict[str, Any]:
        """Return stable, JSON-serializable architecture metadata."""

        return {
            "statistic_groups": list(self.statistic_names),
            "correction_scale": self.correction_scale,
            "gate_parameterization": "shared_scalar",
            "gate_initialization": 0.0,
            "projection_initialization": self.projection_initialization,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        grouped_statistics = self.pool_tokens(tokens).chunk(len(self.statistic_names), dim=-1)
        correction = sum(
            projection(statistic)
            for projection, statistic in zip(self.projections, grouped_statistics)
        )
        return self.linear(mean + self.correction_scale * self.gate * correction)


class TemporalPyramidStatsResidualHead(nn.Module):
    """Mean-linear baseline plus a zero-start low-rank segment correction.

    The final token sequence is split into contiguous, ordered segments.  Each
    segment contributes standard deviation, range, mean absolute deviation,
    and mean absolute value features.  A small ``down``/``up`` factorization
    maps the concatenated statistics back into the embedding dimension.  The
    ``up`` factor starts at zero, so the head is exactly mean-linear at the
    beginning of training while still giving that factor a learning signal.
    """

    statistic_names = ("std", "range", "mad", "mean_abs")

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        segments: int = 2,
        correction_rank: int = 8,
    ):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        if segments < 2:
            raise ValueError("segments must be at least 2")
        if correction_rank <= 0:
            raise ValueError("correction_rank must be positive")
        self.embed_dim = embed_dim
        self.n_outputs = n_outputs
        self.segments = int(segments)
        self.correction_rank = int(correction_rank)
        self.correction_scale = 0.5

        # Construct the baseline first so it has the exact mean-linear RNG
        # initialization.  The low-rank correction starts at zero through its
        # second factor, while ``down`` remains a useful trainable factor.
        self.linear = nn.Linear(embed_dim, n_outputs)
        statistic_dim = len(self.statistic_names) * self.segments * embed_dim
        self.down = nn.Linear(statistic_dim, self.correction_rank, bias=False)
        self.up = nn.Linear(self.correction_rank, embed_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def _segment_statistics(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] < self.segments:
            raise AdapterContractError(
                "temporal_pyramid_stats requires at least as many tokens as segments"
            )
        segment_features = []
        for segment in torch.tensor_split(tokens, self.segments, dim=1):
            mean = segment.mean(dim=1, keepdim=True)
            standard_deviation = segment.std(dim=1, unbiased=False)
            value_range = segment.amax(dim=1) - segment.amin(dim=1)
            mean_absolute_deviation = (segment - mean).abs().mean(dim=1)
            mean_absolute_value = segment.abs().mean(dim=1)
            segment_features.append(
                torch.cat(
                    (
                        standard_deviation,
                        value_range,
                        mean_absolute_deviation,
                        mean_absolute_value,
                    ),
                    dim=-1,
                )
            )
        return torch.cat(segment_features, dim=-1)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return ordered per-segment statistics for the correction branch."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        return self._segment_statistics(tokens)

    def metadata(self) -> dict[str, Any]:
        """Return stable, JSON-serializable architecture metadata."""

        return {
            "segments": self.segments,
            "statistics": list(self.statistic_names),
            "correction_rank": self.correction_rank,
            "correction_scale": self.correction_scale,
            "low_rank_parameterization": "down_then_up",
            "up_initialization": "zero",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        statistics = self._segment_statistics(tokens)
        correction = self.up(self.down(statistics))
        return self.linear(mean + self.correction_scale * correction)


class MeanCovarianceResidualHead(nn.Module):
    """Mean-linear baseline plus a zero-start diagonal covariance correction.

    The covariance contract is intentionally diagonal: each feature contributes
    its sample variance across final tokens.  A small ``down``/``up`` factor-
    ization maps those D variance values back into the embedding dimension.
    ``up`` starts at zero, so initialization is exactly mean-linear.
    """

    covariance_mode = "diagonal"

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        projection_rank: int = 4,
    ):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        if projection_rank <= 0:
            raise ValueError("projection_rank must be positive")
        self.embed_dim = embed_dim
        self.n_outputs = n_outputs
        self.projection_rank = int(projection_rank)
        self.correction_scale = 0.5

        # Construct the baseline first so its initialization matches the
        # mean-linear control.  Zeroing ``up`` preserves exact parity.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.down = nn.Linear(embed_dim, self.projection_rank, bias=False)
        self.up = nn.Linear(self.projection_rank, embed_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return the diagonal sample covariance for final tokens."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        if tokens.shape[1] < 2:
            raise AdapterContractError(
                "mean covariance requires at least two tokens for sample variance"
            )
        return tokens.var(dim=1, unbiased=True)

    def metadata(self) -> dict[str, Any]:
        """Return stable, JSON-serializable architecture metadata."""

        return {
            "covariance_mode": self.covariance_mode,
            "covariance_features": "diagonal_sample_variance",
            "projection_rank": self.projection_rank,
            "correction_scale": self.correction_scale,
            "low_rank_parameterization": "down_then_up",
            "up_initialization": "zero",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        covariance_diagonal = self.pool_tokens(tokens)
        correction = self.up(self.down(covariance_diagonal))
        return self.linear(mean + self.correction_scale * correction)


class MeanStatsAttentionResidualHead(nn.Module):
    """Mean-linear baseline with zero-start attention and statistics corrections."""

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        query_token: torch.Tensor | None = None,
        query_initialization_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        if query_token is None:
            query_token = torch.zeros(1, 1, embed_dim)
            self.query_initialization = "zero_uniform_attention"
        else:
            query_token = _validate_query_token(query_token, embed_dim=embed_dim)
            self.query_initialization = "provided"
        if query_initialization_metadata is not None:
            initialization_name = query_initialization_metadata.get("query_initialization")
            if not isinstance(initialization_name, str):
                raise AdapterContractError("mean_stats_attention_residual query initialization metadata must name its source")
            self.query_initialization = initialization_name

        self.query_token = nn.Parameter(query_token.detach().clone())
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.attention_correction = nn.Linear(embed_dim, n_outputs, bias=False)
        self.stats_correction = nn.Linear(2 * embed_dim, n_outputs, bias=False)
        nn.init.zeros_(self.attention_correction.weight)
        nn.init.zeros_(self.stats_correction.weight)
        self.attention_scale = 0.25
        self.stats_scale = 0.5

    def _features(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = tokens.mean(dim=1)
        query = self.query_token.expand(tokens.shape[0], -1, -1)
        scores = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, tokens).squeeze(1)
        standard_deviation = tokens.std(dim=1, unbiased=False)
        value_range = tokens.amax(dim=1) - tokens.amin(dim=1)
        statistics = torch.cat((standard_deviation, value_range), dim=-1)
        return mean, attention - mean, statistics

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return the mean representation used by the baseline branch."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        return tokens.mean(dim=1)

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean, attention_residual, statistics = self._features(tokens)
        return (
            self.linear(mean)
            + self.attention_scale * self.attention_correction(attention_residual)
            + self.stats_scale * self.stats_correction(statistics)
        )


class MeanAttentionGatedHead(nn.Module):
    """Mean-linear baseline plus a detached, scalar-gated attention correction.

    The baseline branch is always ``Linear(mean(tokens))``.  The correction
    sees detached tokens, so it cannot change the encoder gradient.  A scalar
    ``gamma`` starts at zero, making the initial prediction exactly the
    mean-linear baseline while still allowing backpropagation to learn when
    the correction is useful.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        query_token: torch.Tensor | None = None,
        query_initialization_metadata: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        self.embed_dim = embed_dim
        if query_token is None:
            query_token = torch.zeros(1, 1, embed_dim)
            self.query_initialization = "zero_uniform_attention"
        else:
            query_token = _validate_query_token(query_token, embed_dim=embed_dim)
            self.query_initialization = "provided"
        if query_initialization_metadata is not None:
            initialization_name = query_initialization_metadata.get("query_initialization")
            if not isinstance(initialization_name, str):
                raise AdapterContractError("mean_attention_gated query initialization metadata must name its source")
            self.query_initialization = initialization_name

        self.query_token = nn.Parameter(query_token.detach().clone())
        # Construct the baseline probe first so its initialization matches the
        # mean-linear control. The correction is small but non-zero: gamma=0
        # makes the initial output exact while giving gamma a learning signal.
        self.linear = nn.Linear(embed_dim, n_outputs)
        self.correction = nn.Linear(embed_dim, n_outputs, bias=False)
        nn.init.normal_(self.correction.weight, mean=0.0, std=0.01)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.correction_scale = 0.25

    def _attention_residual(self, tokens: torch.Tensor) -> torch.Tensor:
        detached_tokens = tokens.detach()
        query = self.query_token.expand(detached_tokens.shape[0], -1, -1)
        scores = torch.matmul(query, detached_tokens.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        weights = torch.softmax(scores, dim=-1)
        attention = torch.matmul(weights, detached_tokens).squeeze(1)
        return attention - detached_tokens.mean(dim=1)

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Return the mean representation used by the baseline branch."""

        return _validate_final_tokens(tokens, embed_dim=self.embed_dim).mean(dim=1)

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        correction = self.correction(self._attention_residual(tokens))
        return self.linear(mean) + self.correction_scale * self.gamma * correction


class MeanStatsResidualDetachedHead(MeanStatsResidualHead):
    """Mean-linear head with a statistics correction that cannot alter encoder gradients."""

    def pool_tokens(self, tokens: Any) -> torch.Tensor:
        """Compute the correction statistics without a path back to the encoder."""

        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        return super().pool_tokens(tokens.detach())


class MeanStatsResidualGradientScaledHead(MeanStatsResidualHead):
    """Statistics residual with a controlled gradient into the encoder."""

    def __init__(self, *, embed_dim: int, n_outputs: int, encoder_gradient_scale: float = 0.5):
        super().__init__(embed_dim=embed_dim, n_outputs=n_outputs)
        if not 0.0 <= encoder_gradient_scale <= 1.0:
            raise ValueError("encoder_gradient_scale must be in [0, 1]")
        self.encoder_gradient_scale = float(encoder_gradient_scale)

    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        scaled_mean = mean.detach() + self.encoder_gradient_scale * (mean - mean.detach())
        return self.linear(scaled_mean) + self.correction_scale * self.correction(self.pool_tokens(tokens.detach()))


class MeanStatsProbeScaledHead(MeanStatsResidualHead):
    """Statistics head with a modestly faster probe update."""

    def __init__(
        self,
        *,
        embed_dim: int,
        n_outputs: int,
        encoder_gradient_scale: float = 1.0,
        probe_gradient_scale: float = 2.0,
    ):
        super().__init__(embed_dim=embed_dim, n_outputs=n_outputs)
        if not 0.0 <= encoder_gradient_scale <= 1.0:
            raise ValueError("encoder_gradient_scale must be in [0, 1]")
        if probe_gradient_scale <= 0.0:
            raise ValueError("probe_gradient_scale must be positive")
        self.encoder_gradient_scale = float(encoder_gradient_scale)
        self.probe_gradient_scale = float(probe_gradient_scale)
    def forward(self, tokens: Any) -> torch.Tensor:
        tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)
        mean = tokens.mean(dim=1)
        scaled_mean = mean.detach() + self.encoder_gradient_scale * (mean - mean.detach())
        statistics = self.pool_tokens(tokens.detach())
        linear_weight = self.linear.weight.detach() + self.probe_gradient_scale * (
            self.linear.weight - self.linear.weight.detach()
        )
        linear_bias = self.linear.bias.detach() + self.probe_gradient_scale * (
            self.linear.bias - self.linear.bias.detach()
        )
        correction_weight = self.correction.weight.detach() + self.probe_gradient_scale * (
            self.correction.weight - self.correction.weight.detach()
        )
        linear_output = F.linear(scaled_mean, linear_weight, linear_bias)
        correction_output = F.linear(
            statistics,
            correction_weight,
            self.correction.bias,
        )
        return linear_output + self.correction_scale * correction_output


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
        elif query_token is not None:
            _validate_query_shape(query_token, embed_dim=embed_dim)

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
            self.gate_logit = nn.Parameter(torch.tensor(math.log(LAST_TUNED_INITIAL_ALPHA / (1.0 - LAST_TUNED_INITIAL_ALPHA))))
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
                    raise AdapterContractError("last_tuned query initialization metadata must name its source")
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
                raise AdapterContractError("last_tuned _test_alpha override must be a finite scalar in [0, 1]") from error
        if not torch.isfinite(_test_alpha).all():
            raise AdapterContractError("last_tuned _test_alpha override must be finite")
        if not bool(((_test_alpha >= 0) & (_test_alpha <= 1)).all()):
            raise AdapterContractError("last_tuned _test_alpha override must be in [0, 1]")
        return _test_alpha

    def _pool_with_query_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool final tokens with the official upstream query-attention formula."""

        assert self.query_token is not None
        query = self.query_token.expand(tokens.shape[0], -1, -1)
        scores = torch.matmul(query, tokens.transpose(-1, -2)) / (self.embed_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, tokens).squeeze(1)

    def _pool_last_tuned(
        self,
        tokens: torch.Tensor,
        *,
        alpha_override: float | torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool tokens with tuned attention and its learned mean residual."""

        assert self.query_token is not None
        if not torch.isfinite(tokens).all():
            raise AdapterContractError("last_tuned final token tensor must contain only finite values")
        self._validate_last_tuned_state()

        query = self.query_token.expand(tokens.shape[0], -1, -1)
        scores = torch.einsum("bqd,btd->bqt", query, tokens) / math.sqrt(self.embed_dim)
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
            if alpha_override is None
            else self._test_alpha_override(alpha_override, reference=mean)
        )
        mixed = mean + residual_alpha * (attention - mean)
        if not torch.isfinite(mixed).all():
            raise AdapterContractError("last_tuned mixed residual must be finite")
        return mixed

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

        if self.variant == "last_tuned":
            return self._pool_last_tuned(tokens, alpha_override=_test_alpha)
        return self._pool_with_query_attention(tokens)

    def forward(self, tokens: Any, **kwargs: Any) -> torch.Tensor:
        representation = self.pool_tokens(tokens, **kwargs)
        return self.linear(self.dropout(self.norm(representation)))


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
            raise AdapterContractError("all requires the official NeuralTrain REVE wrapper with a model attribute")

        channel_indices = getattr(self.wrapped_encoder, "channel_indices", None)
        if channel_indices is not None:
            eeg = eeg[:, channel_indices]
        # The official NeuralBench NtReve wrapper resolves its positions
        # internally. Keep ``pos`` in this compatibility signature, but do
        # not forward per-batch channel positions in production.
        del pos
        output = inner(eeg, return_output=True)
        if isinstance(output, torch.Tensor):
            raise AdapterContractError("all requires return_output=True to expose the ordered layer sequence")
        if not isinstance(output, (list, tuple)):
            raise AdapterContractError("all requires return_output=True to return an ordered sequence")
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
        mean_gradient_scale: float = 0.5,
        correction_gradient_scale: float = 1.0,
    ):
        super().__init__()
        if variant in {"mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "mean_rich_stats_gradient_routes", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "multi_query_rich_stats"}:
            validate_local_head_variant(variant)
        elif variant == "last_tuned":
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
        if variant == "mean_linear_copy":
            self.head = MeanLinearCopyHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_linear_detached":
            self.head = MeanLinearDetachedHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_linear_warmup":
            self.head = MeanLinearWarmupHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_linear_gradient_scaled":
            self.head = MeanLinearGradientScaledHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_linear_probe_scaled":
            self.head = MeanLinearProbeScaledHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_anchor":
            self.head = MeanAnchorHead(
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                query_token=query_token,
                query_initialization_metadata=query_initialization_metadata,
            )
        elif variant == "mean_residual":
            self.head = MeanResidualHead(
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                query_token=query_token,
                query_initialization_metadata=query_initialization_metadata,
            )
        elif variant == "mean_vector_anchor":
            self.head = MeanVectorAnchorHead(
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                query_token=query_token,
                query_initialization_metadata=query_initialization_metadata,
            )
        elif variant == "mean_mlp_residual":
            self.head = MeanMLPResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_stats_residual":
            self.head = MeanStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "global_stats_residual":
            self.head = GlobalStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_rich_stats_residual":
            self.head = MeanRichStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_rich_stats_gradient_routes":
            self.head = MeanRichStatsGradientRoutesHead(
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                mean_gradient_scale=mean_gradient_scale,
                correction_gradient_scale=correction_gradient_scale,
            )
        elif variant == "mean_anchor_ensemble":
            self.head = MeanAnchorEnsembleHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant in {"mean_reliability_shrinkage", "mean_reliability_stable"}:
            self.head = MeanReliabilityShrinkageHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "grouped_rich_stats_shrinkage":
            self.head = GroupedRichStatsShrinkageHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "grouped_stats_shared_gate":
            self.head = GroupedStatsSharedGateHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "temporal_pyramid_stats":
            self.head = TemporalPyramidStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_covariance_residual":
            self.head = MeanCovarianceResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "multi_query_rich_stats":
            self.head = MultiQueryRichStatsResidualHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_stats_residual_detached":
            self.head = MeanStatsResidualDetachedHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_stats_residual_gradient_scaled":
            self.head = MeanStatsResidualGradientScaledHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_stats_probe_scaled":
            self.head = MeanStatsProbeScaledHead(embed_dim=embed_dim, n_outputs=n_outputs)
        elif variant == "mean_stats_attention_residual":
            self.head = MeanStatsAttentionResidualHead(
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                query_token=query_token,
                query_initialization_metadata=query_initialization_metadata,
            )
        elif variant == "mean_attention_gated":
            self.head = MeanAttentionGatedHead(
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                query_token=query_token,
                query_initialization_metadata=query_initialization_metadata,
            )
        elif variant == "last_tuned":
            self.head = UpstreamReveHead(
                variant=variant,
                embed_dim=embed_dim,
                n_outputs=n_outputs,
                dropout=dropout,
                query_token=query_token,
                query_initialization_metadata=query_initialization_metadata,
            )
        else:
            self.head = UpstreamReveHead(variant=variant, embed_dim=embed_dim, n_outputs=n_outputs, dropout=dropout)

    def _encode(
        self, eeg: torch.Tensor, channel_positions: torch.Tensor | None
    ) -> Any:
        del channel_positions
        if self.variant in {"mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "mean_rich_stats_gradient_routes", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "multi_query_rich_stats"}:
            # The official NtReve wrapper keeps its resolved REVE positions
            # internally and does not receive per-batch channel positions.
            return self.encoder(eeg)
        if self.variant == "all":
            assert self.all_layer_encoder is not None
            return self.all_layer_encoder(eeg)
        return self.encoder(eeg)

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
    mean_gradient_scale: float = 0.5,
    correction_gradient_scale: float = 1.0,
) -> Any:
    """Create a concrete official NeuralBench wrapper config lazily."""

    if variant in {"mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "mean_rich_stats_gradient_routes", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "multi_query_rich_stats"}:
        validate_local_head_variant(variant)
    elif variant == "last_tuned":
        validate_last_tuned_protocol(variant)
    else:
        validate_upstream_head_variant(variant)
    if dropout != DEFAULT_UPSTREAM_DROPOUT:
        raise ValueError(f"the first upstream-head comparison fixes dropout=0.0; got {dropout}")

    from neuralbench.modules import DownstreamWrapper

    class UpstreamReveHeadWrapper(DownstreamWrapper):
        head_variant: str = variant
        head_dropout: float = float(dropout)
        head_mean_gradient_scale: float = float(mean_gradient_scale)
        head_correction_gradient_scale: float = float(correction_gradient_scale)

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
                raise ProtocolMismatchError("upstream REVE head experiment requires full backbone fine-tuning")

            query_token = None
            query_metadata = None
            channel_positions = None
            if self.head_variant == "last_tuned":
                query_token, query_metadata = initialize_last_tuned_query(model, dummy_batch, provenance=_NEURALBENCH_TRAIN_DUMMY_CONTEXT)
                input_key = _encoder_primary_input_key(model)
                sample = dummy_batch[input_key]
                channel_positions = dummy_batch.get("channel_positions")
            elif self.head_variant in {"mean_anchor", "mean_residual", "mean_vector_anchor", "mean_stats_attention_residual", "mean_attention_gated"}:
                query_token, query_metadata = initialize_mean_anchor_query(model, dummy_batch, provenance=_NEURALBENCH_TRAIN_DUMMY_CONTEXT)
                input_key = _encoder_primary_input_key(model)
                sample = dummy_batch[input_key]
            else:
                input_name = next(iter(dummy_batch))
                sample = dummy_batch[input_name]
            if sample is None:
                raise AdapterContractError("dummy batch contains no EEG tensor")

            if self.head_variant in {"mean_linear_copy", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_linear_probe_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled", "mean_stats_probe_scaled", "mean_stats_attention_residual", "mean_attention_gated", "global_stats_residual", "mean_rich_stats_residual", "mean_rich_stats_gradient_routes", "mean_anchor_ensemble", "mean_reliability_shrinkage", "mean_reliability_stable", "grouped_rich_stats_shrinkage", "grouped_stats_shared_gate", "temporal_pyramid_stats", "mean_covariance_residual", "multi_query_rich_stats"}:
                # Match DownstreamWrapper.build: run the already-built model
                # once before constructing its linear probe.
                with torch.no_grad():
                    model.eval()
                    model(sample)
                    model.train()

            head_model = UpstreamReveHeadModel(
                model,
                variant=self.head_variant,
                n_outputs=n_outputs,
                dropout=self.head_dropout,
                query_token=query_token,
                query_initialization_metadata=query_metadata,
                mean_gradient_scale=self.head_mean_gradient_scale,
                correction_gradient_scale=self.head_correction_gradient_scale,
            )
            with torch.no_grad():
                head_model.eval()
                head_model(sample, channel_positions=channel_positions)
            head_model.train()
            return head_model

    return UpstreamReveHeadWrapper(model_output_key=None, aggregation=None, probe_config=None, head_variant=variant, head_dropout=float(dropout))
