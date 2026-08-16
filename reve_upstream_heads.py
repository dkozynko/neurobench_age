"""Exact upstream REVE downstream heads and the NeuralBench adapter.

The pure head implementation in this module follows the pinned upstream REVE
classifier.  The optional NeuralBench adapter is created lazily so the head
math can be tested without importing the full NeuralBench/NeuralTrain stack.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import platform
import sys
from pathlib import Path
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

HEAD_VARIANTS = ("mean_linear", "last_avg", "last", "all")
UPSTREAM_HEAD_VARIANTS = ("last_avg", "last", "all")
DEFAULT_UPSTREAM_DROPOUT = 0.0
RMS_NORM_EPS = 1e-6
UPSTREAM_HEAD_HIDDEN_SIZE = 512
UPSTREAM_HEAD_INIT_STD = UPSTREAM_HEAD_HIDDEN_SIZE ** -0.5
UPSTREAM_HEAD_INIT_CUTOFF = 3.0


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


def validate_upstream_head_variant(variant: str) -> str:
    """Validate and return a non-baseline upstream head variant."""

    validate_head_variant(variant)
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
    """Upstream REVE mean or query-attention regression head."""

    def __init__(
        self,
        *,
        variant: str,
        embed_dim: int,
        n_outputs: int,
        dropout: float = DEFAULT_UPSTREAM_DROPOUT,
        query_token: torch.Tensor | None = None,
    ):
        super().__init__()
        validate_upstream_head_variant(variant)
        if embed_dim <= 0 or n_outputs <= 0:
            raise ValueError("embed_dim and n_outputs must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if query_token is not None and tuple(query_token.shape) != (1, 1, embed_dim):
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

    def pool_tokens(self, tokens: Any, **kwargs: Any) -> torch.Tensor:
        """Return the representation immediately before RMSNorm/dropout/linear."""

        _reject_mask_kwargs(kwargs)
        if self.variant == "all":
            tokens = concatenate_all_layers(tokens, embed_dim=self.embed_dim)
        else:
            tokens = _validate_final_tokens(tokens, embed_dim=self.embed_dim)

        if self.variant == "last_avg":
            return tokens.mean(dim=1)

        assert self.query_token is not None
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
    ):
        super().__init__()
        validate_upstream_head_variant(variant)
        self.encoder = encoder
        self.variant = variant
        self.all_layer_encoder = AllLayerReveEncoder(encoder) if variant == "all" else None
        # The upstream downstream classifier initializes its own query token;
        # the pretrained NtReve encoder has no classifier token in its state.
        # ``UpstreamReveHead`` performs that explicit torch.randn init after
        # the official experiment seed has been applied.
        embed_dim = _infer_embed_dim(encoder)
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
    "concatenate_all_layers",
    "make_upstream_reve_wrapper",
    "runtime_metadata",
    "source_lock_metadata",
    "validate_head_variant",
    "validate_official_protocol",
    "validate_upstream_head_variant",
    "verify_upstream_source_hashes",
]
