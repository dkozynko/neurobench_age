from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from neurobench_age.pipelines.frozen_probe import (
    FrozenEncoderError,
    RepresentationCacheIdentity,
    assert_frozen_encoder,
    assert_head_only_optimizer,
    build_head_optimizer,
    encoder_state_sha256,
    extract_frozen_representations,
    load_cached_representations,
    load_reve_encoder,
    write_cached_representations,
)


class TinyEncoder(nn.Module):
    def __init__(self, *, mutate_state: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])
        self.dropout = nn.Dropout(0.8)
        self.register_buffer("forward_count", torch.tensor(0))
        self.mutate_state = mutate_state

    def forward(self, value: torch.Tensor, *, return_output: bool = False):
        if self.mutate_state:
            self.forward_count.add_(1)
        outputs = []
        for layer in self.layers:
            value = self.dropout(torch.tanh(layer(value)))
            outputs.append(value)
        return outputs if return_output else outputs[-1]


def _identity(subject_id: str = "sub-001") -> RepresentationCacheIdentity:
    return RepresentationCacheIdentity(
        protocol_sha256="a" * 64,
        checkpoint="brain-bzh/reve-base",
        checkpoint_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        preprocessing_sha256="d" * 64,
        subject_id=subject_id,
        source_tree_sha256="e" * 64,
    )


def _evidence() -> dict[str, object]:
    return {
        "encoder_frozen": True,
        "encoder_eval_mode": True,
        "inference_mode": True,
        "layer_indices": [-2, -1],
        "state_sha256_before": "f" * 64,
        "state_sha256_after": "f" * 64,
    }


def test_extraction_freezes_encoder_uses_eval_and_inference_mode() -> None:
    torch.manual_seed(5)
    encoder = TinyEncoder()
    encoder.train()
    windows = torch.randn(2, 6, 4, requires_grad=True)

    representations, evidence = extract_frozen_representations(
        encoder, windows, layer_indices=(-2, -1)
    )

    assert encoder.training is False
    assert all(parameter.requires_grad is False for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert set(representations) == {-2, -1}
    assert all(tensor.requires_grad is False for tensor in representations.values())
    assert evidence["inference_mode"] is True
    assert evidence["state_sha256_before"] == evidence["state_sha256_after"]
    assert_frozen_encoder(encoder, expected_state_sha256=evidence["state_sha256_before"])


def test_extraction_is_repeatable_even_when_encoder_contains_dropout() -> None:
    torch.manual_seed(6)
    encoder = TinyEncoder()
    windows = torch.randn(2, 6, 4)

    first, _ = extract_frozen_representations(encoder, windows, layer_indices=(-2, -1))
    second, _ = extract_frozen_representations(encoder, windows, layer_indices=(-2, -1))

    assert torch.equal(first[-2], second[-2])
    assert torch.equal(first[-1], second[-1])


def test_extraction_detects_encoder_state_mutation() -> None:
    encoder = TinyEncoder(mutate_state=True)

    with pytest.raises(FrozenEncoderError, match="state changed"):
        extract_frozen_representations(
            encoder, torch.randn(2, 6, 4), layer_indices=(-2, -1)
        )


def test_head_optimizer_contains_no_encoder_parameters() -> None:
    encoder = TinyEncoder()
    head = nn.Linear(4, 1)

    optimizer = build_head_optimizer(
        head, encoder=encoder, learning_rate=1e-3, weight_decay=1e-4
    )

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in head.parameters()}
    assert not optimizer_ids & {id(parameter) for parameter in encoder.parameters()}


def test_optimizer_audit_rejects_encoder_parameter_ownership() -> None:
    encoder = TinyEncoder()
    head = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(
        [*head.parameters(), *encoder.parameters()], lr=1e-3
    )

    with pytest.raises(FrozenEncoderError, match="encoder parameters"):
        assert_head_only_optimizer(optimizer, head=head, encoder=encoder)


def test_assert_frozen_encoder_detects_train_mode_gradients_and_state_drift() -> None:
    encoder = TinyEncoder()
    expected = encoder_state_sha256(encoder)
    encoder.train()
    with pytest.raises(FrozenEncoderError, match="eval mode"):
        assert_frozen_encoder(encoder, expected_state_sha256=expected)

    encoder.eval()
    next(encoder.parameters()).requires_grad_(True)
    with pytest.raises(FrozenEncoderError, match="requires_grad"):
        assert_frozen_encoder(encoder, expected_state_sha256=expected)


def test_cache_round_trip_rejects_identity_drift_and_missing_layers(tmp_path: Path) -> None:
    identity = _identity()
    representations = {-2: torch.randn(3, 4), -1: torch.randn(3, 4)}

    write_cached_representations(
        tmp_path, identity, representations, evidence=_evidence()
    )
    loaded = load_cached_representations(
        tmp_path, identity, required_layers=(-2, -1)
    )

    assert torch.equal(loaded[-2], representations[-2])
    assert torch.equal(loaded[-1], representations[-1])
    with pytest.raises(FrozenEncoderError, match="cache entry"):
        load_cached_representations(
            tmp_path, replace(identity, subject_id="sub-002"), required_layers=(-2, -1)
        )
    with pytest.raises(FrozenEncoderError, match="required layers"):
        load_cached_representations(tmp_path, identity, required_layers=(-3, -1))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.update({"status": "writing"}), "completion"),
        (lambda metadata: metadata.update({"layers": [-3, -2, -1]}), "layer inventory"),
        (lambda metadata: metadata.update({"evidence": {}}), "evidence"),
    ],
)
def test_cache_rejects_incomplete_or_tampered_metadata(
    tmp_path: Path, mutation, message: str
) -> None:
    identity = _identity()
    entry = write_cached_representations(
        tmp_path,
        identity,
        {-2: torch.randn(3, 4), -1: torch.randn(3, 4)},
        evidence=_evidence(),
    )
    metadata_path = entry / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutation(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(FrozenEncoderError, match=message):
        load_cached_representations(tmp_path, identity)


def test_cache_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    identity = _identity()
    entry = write_cached_representations(
        tmp_path,
        identity,
        {-2: torch.randn(3, 4), -1: torch.randn(3, 4)},
        evidence=_evidence(),
    )
    with (entry / "representations.pt").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(FrozenEncoderError, match="payload hash"):
        load_cached_representations(tmp_path, identity)


def test_cache_writer_requires_complete_frozen_extraction_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(FrozenEncoderError, match="evidence"):
        write_cached_representations(
            tmp_path,
            _identity(),
            {-2: torch.randn(3, 4), -1: torch.randn(3, 4)},
            evidence={
                "state_sha256_before": "f" * 64,
                "state_sha256_after": "f" * 64,
            },
        )


def test_reve_loader_accepts_only_predeclared_checkpoint_and_freezes_result() -> None:
    built = TinyEncoder()

    loaded = load_reve_encoder(
        "brain-bzh/reve-base",
        channel_names=("Cz", "Fz"),
        mapping_path=Path("mapping.json"),
        loader=lambda **kwargs: built,
    )

    assert loaded is built
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(FrozenEncoderError, match="checkpoint"):
        load_reve_encoder(
            "other/model",
            channel_names=("Cz",),
            mapping_path=None,
            loader=lambda **kwargs: TinyEncoder(),
        )
