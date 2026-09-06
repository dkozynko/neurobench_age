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
from neurobench_age.pipelines.frozen_probe_training import (
    APPROVED_HEADS,
    CachedSubjectRecord,
    audit_frozen_probe_inventory,
    build_frozen_probe_head,
    predict_cached_subjects,
    required_layer_for_head,
    train_frozen_probe_run,
    train_frozen_probe_study,
    validate_training_records,
)
from neurobench_age.research.protocol import TrainingContract


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


@pytest.mark.parametrize(
    ("head_name", "expected_layer"),
    [
        ("mean_linear", -1),
        ("mean_layer_linear", -2),
        ("mean_rich_stats_residual", -1),
        ("multi_query_rich_stats", -1),
    ],
)
def test_predeclared_heads_consume_one_declared_cached_layer(
    head_name: str, expected_layer: int
) -> None:
    torch.manual_seed(33)
    head = build_frozen_probe_head(head_name, embed_dim=4)

    prediction = head(torch.randn(3, 6, 4))

    assert APPROVED_HEADS == (
        "mean_linear",
        "mean_layer_linear",
        "mean_rich_stats_residual",
        "multi_query_rich_stats",
    )
    assert required_layer_for_head(head_name) == expected_layer
    assert tuple(prediction.shape) == (3, 1)
    assert not hasattr(head, "encoder")


def test_frozen_probe_rejects_unapproved_head() -> None:
    with pytest.raises(FrozenEncoderError, match="approved heads"):
        build_frozen_probe_head("mean_mlp_residual", embed_dim=4)
    with pytest.raises(FrozenEncoderError, match="approved heads"):
        required_layer_for_head("mean_mlp_residual")


def test_training_records_reject_test_split_overlap_and_non_finite_age() -> None:
    with pytest.raises(FrozenEncoderError, match="train or validation"):
        CachedSubjectRecord(
            subject_id="sub-test",
            split="test",
            age=12.0,
            cache_identity=_identity("sub-test"),
        )
    with pytest.raises(FrozenEncoderError, match="finite age"):
        CachedSubjectRecord(
            subject_id="sub-nan",
            split="train",
            age=float("nan"),
            cache_identity=_identity("sub-nan"),
        )

    train = CachedSubjectRecord("sub-001", "train", 10.0, _identity("sub-001"))
    validation = CachedSubjectRecord(
        "sub-001", "validation", 10.0, _identity("sub-001")
    )
    with pytest.raises(FrozenEncoderError, match="duplicate or overlapping"):
        validate_training_records((train, validation))


def test_validation_predictions_average_windows_per_subject(tmp_path: Path) -> None:
    records = []
    for subject_id, age, window_values in (
        ("sub-001", 2.0, (1.0, 3.0)),
        ("sub-002", 5.0, (4.0, 6.0)),
    ):
        identity = _identity(subject_id)
        final = torch.tensor(window_values).reshape(2, 1, 1)
        write_cached_representations(
            tmp_path,
            identity,
            {-2: final + 100.0, -1: final},
            evidence=_evidence(),
        )
        records.append(
            CachedSubjectRecord(subject_id, "validation", age, identity)
        )
    head = build_frozen_probe_head("mean_linear", embed_dim=1)
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()

    predictions = predict_cached_subjects(
        head,
        records,
        cache_root=tmp_path,
        head_name="mean_linear",
        batch_size=1,
        device="cpu",
    )

    assert predictions == (
        {"subject_id": "sub-001", "age": 2.0, "prediction": 2.0},
        {"subject_id": "sub-002", "age": 5.0, "prediction": 5.0},
    )


def _tiny_training_records(cache_root: Path) -> tuple[CachedSubjectRecord, ...]:
    records = []
    for index, (split, age) in enumerate(
        (
            ("train", 8.0),
            ("train", 11.0),
            ("train", 14.0),
            ("validation", 9.0),
            ("validation", 12.0),
            ("validation", 15.0),
        ),
        start=1,
    ):
        subject_id = f"sub-{index:03d}"
        identity = _identity(subject_id)
        signal = torch.full((3, 2, 2), age / 10.0)
        write_cached_representations(
            cache_root,
            identity,
            {-2: signal + 0.5, -1: signal},
            evidence=_evidence(),
        )
        records.append(CachedSubjectRecord(subject_id, split, age, identity))
    return tuple(records)


def _tiny_training_contract() -> TrainingContract:
    return TrainingContract(
        seeds=tuple(range(33, 43)),
        optimizer="AdamW",
        learning_rate=0.05,
        weight_decay=0.0001,
        batch_size=2,
        max_epochs=5,
        patience=2,
        loss="MSELoss",
        checkpoint_metric="validation_pearson",
        metric_mode="max",
    )


def test_training_run_is_validation_only_auditable_and_exactly_resumable(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    records = _tiny_training_records(cache_root)
    run_dir = tmp_path / "runs" / "mean_linear" / "seed-33"

    first = train_frozen_probe_run(
        head_name="mean_linear",
        seed=33,
        records=records,
        cache_root=cache_root,
        run_dir=run_dir,
        training=_tiny_training_contract(),
        device="cpu",
    )

    assert first.reused is False
    manifest = first.manifest
    assert manifest["status"] == "complete"
    assert manifest["head_name"] == "mean_linear"
    assert manifest["seed"] == 33
    assert manifest["run_identity"]["cache_contract"] == {
        "protocol_sha256": "a" * 64,
        "checkpoint": "brain-bzh/reve-base",
        "checkpoint_sha256": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "preprocessing_sha256": "d" * 64,
        "source_tree_sha256": "e" * 64,
    }
    assert manifest["checkpoint_selection"] == {
        "metric": "validation_pearson",
        "mode": "max",
        "tie_break": "earliest_epoch",
    }
    assert 1 <= manifest["selected_epoch"] <= 5
    assert manifest["head_parameters"]["trainable"] > 0
    assert manifest["head_parameters"]["total"] == manifest["head_parameters"]["trainable"]
    assert manifest["optimizer"]["name"] == "AdamW"
    assert len(manifest["validation_history"]) >= 1
    assert all("test" not in key for row in manifest["validation_history"] for key in row)
    assert manifest["runtime_seconds"] >= 0.0
    assert manifest["peak_process_rss_bytes"] > 0
    checkpoint_path = run_dir / "head_checkpoint.pt"
    before = checkpoint_path.read_bytes()

    second = train_frozen_probe_run(
        head_name="mean_linear",
        seed=33,
        records=records,
        cache_root=cache_root,
        run_dir=run_dir,
        training=_tiny_training_contract(),
        device="cpu",
    )

    assert second.reused is True
    assert second.manifest == first.manifest
    assert checkpoint_path.read_bytes() == before

    with checkpoint_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(FrozenEncoderError, match="checkpoint hash"):
        train_frozen_probe_run(
            head_name="mean_linear",
            seed=33,
            records=records,
            cache_root=cache_root,
            run_dir=run_dir,
            training=_tiny_training_contract(),
            device="cpu",
        )


def test_training_rejects_mixed_cache_provenance_before_fitting(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    records = list(_tiny_training_records(cache_root))
    changed_identity = replace(
        records[-1].cache_identity, source_tree_sha256="9" * 64
    )
    records[-1] = replace(records[-1], cache_identity=changed_identity)

    with pytest.raises(FrozenEncoderError, match="cache provenance"):
        train_frozen_probe_run(
            head_name="mean_linear",
            seed=33,
            records=records,
            cache_root=cache_root,
            run_dir=tmp_path / "run",
            training=_tiny_training_contract(),
            device="cpu",
        )


def test_study_requires_exact_hash_valid_forty_run_inventory(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    records = _tiny_training_records(cache_root)
    output_root = tmp_path / "runs"
    training = replace(
        _tiny_training_contract(), max_epochs=1, patience=1
    )

    inventory = train_frozen_probe_study(
        records=records,
        cache_root=cache_root,
        output_root=output_root,
        training=training,
        device="cpu",
    )

    assert inventory["status"] == "complete"
    assert inventory["run_count"] == 40
    assert len(inventory["runs"]) == 40
    assert {
        (run["head_name"], run["seed"]) for run in inventory["runs"]
    } == {
        (head_name, seed)
        for head_name in APPROVED_HEADS
        for seed in range(33, 43)
    }
    assert audit_frozen_probe_inventory(
        output_root=output_root,
        records=records,
        training=training,
    ) == inventory

    extra = output_root / "unexpected-head"
    extra.mkdir()
    with pytest.raises(FrozenEncoderError, match="exact 40-run inventory"):
        audit_frozen_probe_inventory(
            output_root=output_root,
            records=records,
            training=training,
        )
    audit_frozen_probe_inventory,
