"""Frozen-encoder representation extraction and cache integrity helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn


PREDECLARED_CHECKPOINT = "brain-bzh/reve-base"
PREDECLARED_LAYERS = (-2, -1)


class FrozenEncoderError(RuntimeError):
    """Raised when frozen extraction or cache evidence is unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_extraction_evidence(evidence: object) -> None:
    if not isinstance(evidence, Mapping):
        raise FrozenEncoderError("cache extraction evidence must be a mapping")
    required_flags = {
        "encoder_frozen": True,
        "encoder_eval_mode": True,
        "inference_mode": True,
    }
    for name, expected in required_flags.items():
        if evidence.get(name) is not expected:
            raise FrozenEncoderError(
                f"cache extraction evidence requires {name}={expected}"
            )
    if tuple(evidence.get("layer_indices", ())) != PREDECLARED_LAYERS:
        raise FrozenEncoderError(
            f"cache extraction evidence requires layers {PREDECLARED_LAYERS}"
        )
    state_before = evidence.get("state_sha256_before")
    state_after = evidence.get("state_sha256_after")
    if (
        not isinstance(state_before, str)
        or not _is_sha256(state_before)
        or not isinstance(state_after, str)
        or not _is_sha256(state_after)
    ):
        raise FrozenEncoderError("cache extraction evidence requires state SHA-256 hashes")
    if state_before != state_after:
        raise FrozenEncoderError("cache extraction evidence reports encoder state drift")


def encoder_state_sha256(encoder: nn.Module) -> str:
    """Hash every named parameter and persistent buffer deterministically."""

    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        if value.numel():
            bytes_view = value.reshape(-1).view(torch.uint8)
            digest.update(bytes_view.numpy().tobytes())
    return digest.hexdigest()


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    if not isinstance(encoder, nn.Module):
        raise FrozenEncoderError("encoder must be a torch.nn.Module")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    encoder.eval()
    return encoder


def assert_frozen_encoder(
    encoder: nn.Module, *, expected_state_sha256: str | None = None
) -> None:
    if encoder.training:
        raise FrozenEncoderError("frozen encoder must remain in eval mode")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise FrozenEncoderError("frozen encoder parameter has requires_grad=True")
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise FrozenEncoderError("frozen encoder accumulated a gradient")
    if expected_state_sha256 is not None:
        actual = encoder_state_sha256(encoder)
        if actual != expected_state_sha256:
            raise FrozenEncoderError(
                "frozen encoder state changed: "
                f"expected={expected_state_sha256} actual={actual}"
            )


def _forward_all_layers(encoder: nn.Module, windows: torch.Tensor) -> Sequence[torch.Tensor]:
    wrapped = getattr(encoder, "model", None)
    if isinstance(wrapped, nn.Module):
        channel_indices = getattr(encoder, "channel_indices", None)
        if channel_indices is not None:
            windows = windows[:, channel_indices]
        output = wrapped(windows, return_output=True)
    else:
        try:
            output = encoder(windows, return_output=True)
        except TypeError as error:
            raise FrozenEncoderError(
                "encoder must expose ordered layers through return_output=True"
            ) from error
    if not isinstance(output, (list, tuple)) or not output:
        raise FrozenEncoderError("encoder did not return a non-empty ordered layer sequence")
    if any(not isinstance(layer, torch.Tensor) for layer in output):
        raise FrozenEncoderError("encoder layer sequence contains a non-tensor value")
    return output


def extract_frozen_representations(
    encoder: nn.Module,
    windows: torch.Tensor,
    *,
    layer_indices: Sequence[int] = PREDECLARED_LAYERS,
    forward_all_layers: Callable[[nn.Module, torch.Tensor], Sequence[torch.Tensor]] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Extract declared layers once while proving the encoder stayed frozen."""

    requested = tuple(int(index) for index in layer_indices)
    if requested != PREDECLARED_LAYERS:
        raise FrozenEncoderError(
            f"frozen extraction requires exactly layers {PREDECLARED_LAYERS}"
        )
    if not isinstance(windows, torch.Tensor) or windows.ndim < 2:
        raise FrozenEncoderError("windows must be a batched torch tensor")
    freeze_encoder(encoder)
    state_before = encoder_state_sha256(encoder)
    forward = forward_all_layers or _forward_all_layers
    with torch.inference_mode():
        all_layers = forward(encoder, windows)
        representations: dict[int, torch.Tensor] = {}
        for index in requested:
            try:
                layer = all_layers[index]
            except IndexError as error:
                raise FrozenEncoderError(
                    f"encoder does not expose required layer {index}"
                ) from error
            representations[index] = layer.detach().cpu().contiguous()
    state_after = encoder_state_sha256(encoder)
    if state_after != state_before:
        raise FrozenEncoderError(
            f"frozen encoder state changed: before={state_before} after={state_after}"
        )
    assert_frozen_encoder(encoder, expected_state_sha256=state_before)
    return representations, {
        "encoder_frozen": True,
        "encoder_eval_mode": True,
        "inference_mode": True,
        "layer_indices": list(requested),
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
    }


def assert_head_only_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    head: nn.Module,
    encoder: nn.Module,
) -> None:
    """Prove that an optimizer owns every trainable head parameter and no encoder parameter."""

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    encoder_ids = {id(parameter) for parameter in encoder.parameters()}
    if optimizer_ids & encoder_ids:
        raise FrozenEncoderError("head optimizer contains encoder parameters")
    expected_head_ids = {
        id(parameter) for parameter in head.parameters() if parameter.requires_grad
    }
    if not expected_head_ids:
        raise FrozenEncoderError("head exposes no trainable parameters")
    if optimizer_ids != expected_head_ids:
        raise FrozenEncoderError(
            "optimizer ownership differs from trainable head parameters"
        )


def build_head_optimizer(
    head: nn.Module,
    *,
    encoder: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create an AdamW optimizer that owns head parameters and nothing else."""

    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise FrozenEncoderError("learning_rate must be finite and positive")
    if not math.isfinite(float(weight_decay)) or weight_decay < 0:
        raise FrozenEncoderError("weight_decay must be finite and non-negative")
    freeze_encoder(encoder)
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if not head_parameters:
        raise FrozenEncoderError("head exposes no trainable parameters")
    optimizer = torch.optim.AdamW(
        head_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    assert_head_only_optimizer(optimizer, head=head, encoder=encoder)
    return optimizer


def load_reve_encoder(
    checkpoint: str,
    *,
    channel_names: Sequence[str],
    mapping_path: Path | None,
    loader: Callable[..., nn.Module] | None = None,
) -> nn.Module:
    """Build the existing REVE backbone and immediately freeze it."""

    if checkpoint != PREDECLARED_CHECKPOINT:
        raise FrozenEncoderError(
            f"checkpoint must be the predeclared {PREDECLARED_CHECKPOINT!r}"
        )
    if loader is None:
        from .independent import load_reve_backbone

        loader = load_reve_backbone
    encoder = loader(
        channel_names=tuple(channel_names),
        mapping_path=mapping_path,
        pretrained_name=checkpoint,
    )
    if not isinstance(encoder, nn.Module):
        raise FrozenEncoderError("REVE loader did not return a torch module")
    return freeze_encoder(encoder)


@dataclass(frozen=True)
class RepresentationCacheIdentity:
    protocol_sha256: str
    checkpoint: str
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    preprocessing_sha256: str
    subject_id: str
    source_tree_sha256: str

    def __post_init__(self) -> None:
        if self.checkpoint != PREDECLARED_CHECKPOINT:
            raise FrozenEncoderError("cache identity has an unexpected checkpoint")
        for field in (
            "protocol_sha256",
            "checkpoint_sha256",
            "dataset_manifest_sha256",
            "preprocessing_sha256",
            "source_tree_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise FrozenEncoderError(f"cache identity {field} is not a SHA-256 digest")
        if not self.subject_id.strip():
            raise FrozenEncoderError("cache identity subject_id is empty")

    @property
    def key(self) -> str:
        return _canonical_sha256(asdict(self))


def write_cached_representations(
    cache_root: Path,
    identity: RepresentationCacheIdentity,
    representations: Mapping[int, torch.Tensor],
    *,
    evidence: Mapping[str, Any],
) -> Path:
    """Publish one complete cache entry; existing entries are immutable."""

    layers = tuple(sorted(int(index) for index in representations))
    if layers != PREDECLARED_LAYERS:
        raise FrozenEncoderError(
            f"cache must contain exactly required layers {PREDECLARED_LAYERS}"
        )
    tensors: dict[int, torch.Tensor] = {}
    for index in layers:
        tensor = representations[index]
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise FrozenEncoderError(f"cache layer {index} must be a finite tensor")
        tensors[index] = tensor.detach().cpu().contiguous()
    _validate_extraction_evidence(evidence)

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    final_dir = cache_root / identity.key
    if final_dir.exists():
        raise FrozenEncoderError(f"cache entry already exists: {final_dir}")
    transaction_dir = Path(tempfile.mkdtemp(prefix=".cache-transaction-", dir=cache_root))
    try:
        payload_path = transaction_dir / "representations.pt"
        torch.save(tensors, payload_path)
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "cache_key": identity.key,
            "identity": asdict(identity),
            "layers": list(layers),
            "tensor_metadata": {
                str(index): {
                    "shape": list(tensors[index].shape),
                    "dtype": str(tensors[index].dtype),
                }
                for index in layers
            },
            "evidence": dict(evidence),
            "payload_sha256": _sha256_file(payload_path),
        }
        (transaction_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        transaction_dir.replace(final_dir)
    except Exception:
        shutil.rmtree(transaction_dir, ignore_errors=True)
        raise
    return final_dir


def load_cached_representations(
    cache_root: Path,
    identity: RepresentationCacheIdentity,
    *,
    required_layers: Sequence[int] = PREDECLARED_LAYERS,
) -> dict[int, torch.Tensor]:
    """Load only a complete cache entry with exact provenance and layer inventory."""

    entry = Path(cache_root) / identity.key
    metadata_path = entry / "metadata.json"
    payload_path = entry / "representations.pt"
    if not metadata_path.is_file() or not payload_path.is_file():
        raise FrozenEncoderError(f"cache entry is missing or incomplete: {entry}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenEncoderError(f"cache entry metadata is invalid: {entry}") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("status") != "complete"
        or metadata.get("cache_key") != identity.key
        or metadata.get("identity") != asdict(identity)
    ):
        raise FrozenEncoderError(f"cache entry identity or completion marker is invalid: {entry}")
    available = tuple(metadata.get("layers", ()))
    if available != PREDECLARED_LAYERS:
        raise FrozenEncoderError(
            "cache layer inventory is invalid: "
            f"expected={PREDECLARED_LAYERS} available={available}"
        )
    _validate_extraction_evidence(metadata.get("evidence"))
    required = tuple(int(index) for index in required_layers)
    if any(index not in available for index in required):
        raise FrozenEncoderError(
            f"cache does not contain required layers: required={required} available={available}"
        )
    if metadata.get("payload_sha256") != _sha256_file(payload_path):
        raise FrozenEncoderError(f"cache entry payload hash does not match: {entry}")
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise FrozenEncoderError(f"cache entry payload is unreadable: {entry}") from error
    if not isinstance(payload, dict):
        raise FrozenEncoderError(f"cache entry payload is not a layer mapping: {entry}")
    result: dict[int, torch.Tensor] = {}
    for index in required:
        tensor = payload.get(index)
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise FrozenEncoderError(f"cache layer {index} is missing or non-finite")
        expected = metadata.get("tensor_metadata", {}).get(str(index), {})
        if list(tensor.shape) != expected.get("shape") or str(tensor.dtype) != expected.get("dtype"):
            raise FrozenEncoderError(f"cache layer {index} metadata does not match payload")
        result[index] = tensor
    return result
