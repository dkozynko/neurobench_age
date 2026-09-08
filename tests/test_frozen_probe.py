from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import neurobench_age.pipelines.frozen_probe_training as training_module

from neurobench_age.pipelines.frozen_probe import (
    CachedTensorMetadata,
    FrozenEncoderError,
    RepresentationCacheIdentity,
    assert_frozen_encoder,
    assert_head_only_optimizer,
    build_head_optimizer,
    encoder_state_sha256,
    extract_frozen_representations,
    inspect_cached_representation_metadata,
    load_cached_representations,
    load_reve_encoder,
    write_cached_representations,
)
from neurobench_age.pipelines.frozen_probe_training import (
    APPROVED_HEADS,
    CachedSubjectRecord,
    FrozenProbeRunResult,
    GlobalWindowBatchPlan,
    ValidatedRepresentationStore,
    audit_frozen_probe_inventory,
    build_frozen_probe_head,
    predict_cached_subjects,
    required_layer_for_head,
    train_frozen_probe_run,
    train_frozen_probe_study,
    validate_training_records,
)
from neurobench_age.research.training_protocol import (
    FrozenProbeTrainingProtocol,
    load_frozen_probe_training_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PROTOCOL_PATH = (
    ROOT / "configs" / "research" / "neuralbench_frozen_probe_training.json"
)
TRAINING_SOURCE_SHA256 = "1" * 64


def test_global_window_batches_cover_every_window_and_cross_subjects() -> None:
    records = (
        CachedSubjectRecord("sub-a", "train", 8.0, _identity("sub-a")),
        CachedSubjectRecord("sub-b", "train", 11.0, _identity("sub-b")),
        CachedSubjectRecord("sub-c", "train", 14.0, _identity("sub-c")),
    )
    tensors = {
        "sub-a": torch.tensor([[[10.0]], [[11.0]]]),
        "sub-b": torch.tensor([[[20.0]], [[21.0]], [[22.0]]]),
        "sub-c": torch.tensor([[[30.0]]]),
    }
    plan = GlobalWindowBatchPlan.build(records, tensors=tensors, batch_size=4)

    first = plan.permuted_batches(
        generator=torch.Generator(device="cpu").manual_seed(33)
    )
    repeated = plan.permuted_batches(
        generator=torch.Generator(device="cpu").manual_seed(33)
    )

    assert first == repeated
    assert plan.total_windows == 6
    assert plan.steps_per_epoch == 2
    assert [len(batch) for batch in first] == [4, 2]
    assert sorted(reference for batch in first for reference in batch) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
    ]
    assert any(len({subject for subject, _ in batch}) > 1 for batch in first)


def test_global_window_batch_materialization_preserves_permutation_order() -> None:
    records = (
        CachedSubjectRecord("sub-a", "train", 8.0, _identity("sub-a")),
        CachedSubjectRecord("sub-b", "train", 11.0, _identity("sub-b")),
    )
    tensors = {
        "sub-a": torch.tensor([[[10.0]], [[11.0]]]),
        "sub-b": torch.tensor([[[20.0]], [[21.0]]]),
    }
    plan = GlobalWindowBatchPlan.build(records, tensors=tensors, batch_size=3)
    references = ((1, 1), (0, 0), (1, 0))

    batch, targets = plan.materialize(references, tensors=tensors)

    assert batch.tolist() == [[[21.0]], [[10.0]], [[20.0]]]
    assert targets.tolist() == [11.0, 8.0, 11.0]


@pytest.mark.parametrize(
    "tensors",
    [
        {},
        {"sub-a": torch.empty(0, 1, 1)},
        {"sub-a": torch.ones(2, 1)},
    ],
)
def test_global_window_plan_rejects_missing_empty_or_invalid_tensors(
    tensors: dict[str, torch.Tensor],
) -> None:
    records = (CachedSubjectRecord("sub-a", "train", 8.0, _identity("sub-a")),)

    with pytest.raises(FrozenEncoderError, match="global window"):
        GlobalWindowBatchPlan.build(records, tensors=tensors, batch_size=2)


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


def test_cache_metadata_inspection_reports_exact_declared_bytes_without_loading_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    tensors = {
        -2: torch.randn(3, 4, 5, dtype=torch.float32),
        -1: torch.randn(3, 4, 5, dtype=torch.float64),
    }
    write_cached_representations(tmp_path, identity, tensors, evidence=_evidence())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("payload loaded"))

    metadata = inspect_cached_representation_metadata(tmp_path, identity)

    assert tuple(metadata) == (-2, -1)
    assert metadata[-2].shape == (3, 4, 5)
    assert metadata[-2].dtype == torch.float32
    assert metadata[-2].nbytes == tensors[-2].numel() * tensors[-2].element_size()
    assert metadata[-1].nbytes == tensors[-1].numel() * tensors[-1].element_size()


def test_representation_store_loads_both_layers_once_and_preserves_tensor_objects(
    tmp_path: Path,
) -> None:
    records = _tiny_training_records(tmp_path)
    loaded_by_subject: dict[str, dict[int, torch.Tensor]] = {}
    calls: list[tuple[str, tuple[int, ...]]] = []

    def strict_loader(cache_root, identity, *, required_layers):
        calls.append((identity.subject_id, tuple(required_layers)))
        loaded = load_cached_representations(
            cache_root, identity, required_layers=required_layers
        )
        loaded_by_subject[identity.subject_id] = loaded
        return loaded

    store = ValidatedRepresentationStore.build(
        records=records,
        cache_root=tmp_path,
        available_memory_bytes=16 * 1024**3,
        strict_loader=strict_loader,
    )

    assert calls == [(record.subject_id, (-2, -1)) for record in records]
    assert store.projected_bytes == store.actual_retained_bytes
    assert store.subject_count == len(records)
    for record in records:
        for layer in (-2, -1):
            assert store.tensor(record.subject_id, layer) is loaded_by_subject[
                record.subject_id
            ][layer]
    with pytest.raises(FrozenEncoderError, match="absent subject"):
        store.tensor("sub-absent", -1)
    with pytest.raises(FrozenEncoderError, match="absent layer"):
        store.tensor(records[0].subject_id, -3)


def test_representation_store_rejects_insufficient_memory_before_loading(
    tmp_path: Path,
) -> None:
    records = _tiny_training_records(tmp_path)
    calls = 0

    def strict_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not load")

    projected = sum(
        tensor.nbytes
        for record in records
        for tensor in inspect_cached_representation_metadata(
            tmp_path, record.cache_identity
        ).values()
    )
    with pytest.raises(FrozenEncoderError, match="insufficient available RAM"):
        ValidatedRepresentationStore.build(
            records=records,
            cache_root=tmp_path,
            available_memory_bytes=projected + 8 * 1024**3 - 1,
            strict_loader=strict_loader,
        )
    assert calls == 0


def test_available_memory_provider_supports_linux_macos_and_rejects_invalid(
    tmp_path: Path,
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1000 kB\nMemAvailable: 123 kB\n")
    assert training_module._available_memory_bytes(
        platform_name="linux", proc_meminfo_path=meminfo
    ) == 123 * 1024

    vm_stat = (
        "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
        "Pages free: 10.\nPages inactive: 20.\nPages speculative: 3.\n"
    )
    assert training_module._available_memory_bytes(
        platform_name="darwin", vm_stat_output=vm_stat
    ) == 33 * 4096

    meminfo.write_text("MemTotal: 1000 kB\n")
    with pytest.raises(FrozenEncoderError, match="available host memory"):
        training_module._available_memory_bytes(
            platform_name="linux", proc_meminfo_path=meminfo
        )


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
        initialization_seed=0,
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
            initialization_seed=0,
            loader=lambda **kwargs: TinyEncoder(),
        )


def test_reve_loader_has_deterministic_full_state_identity() -> None:
    torch.manual_seed(123)
    first = load_reve_encoder(
        "brain-bzh/reve-base",
        channel_names=("Cz", "Fz"),
        mapping_path=Path("mapping.json"),
        initialization_seed=0,
        loader=lambda **kwargs: TinyEncoder(),
    )
    torch.manual_seed(456)
    second = load_reve_encoder(
        "brain-bzh/reve-base",
        channel_names=("Cz", "Fz"),
        mapping_path=Path("mapping.json"),
        initialization_seed=0,
        loader=lambda **kwargs: TinyEncoder(),
    )

    assert encoder_state_sha256(first) == encoder_state_sha256(second)


def test_reve_loader_restores_callers_torch_rng_state() -> None:
    torch.manual_seed(991)
    expected_next_value = torch.rand(4)

    torch.manual_seed(991)
    load_reve_encoder(
        "brain-bzh/reve-base",
        channel_names=("Cz", "Fz"),
        mapping_path=Path("mapping.json"),
        initialization_seed=0,
        loader=lambda **kwargs: TinyEncoder(),
    )

    assert torch.equal(torch.rand(4), expected_next_value)


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


def _tiny_training_contract() -> FrozenProbeTrainingProtocol:
    protocol = load_frozen_probe_training_protocol(TRAINING_PROTOCOL_PATH)
    return replace(
        protocol,
        optimizer=replace(
            protocol.optimizer,
            learning_rate=0.05,
            weight_decay=0.0001,
        ),
        scheduler=replace(protocol.scheduler, max_learning_rate=0.05),
        batch_size=2,
        max_epochs=5,
        patience=2,
        sha256="2" * 64,
    )


def test_training_run_uses_one_cycle_global_batches_and_gradient_clipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    records = _tiny_training_records(cache_root)
    training = replace(_tiny_training_contract(), max_epochs=1, patience=1)
    clipping_calls: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def traced_clip(parameters, max_norm, *args, **kwargs):
        clipping_calls.append(float(max_norm))
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", traced_clip)

    result = train_frozen_probe_run(
        head_name="mean_linear",
        seed=33,
        records=records,
        cache_root=cache_root,
        run_dir=tmp_path / "run",
        training=training,
        device="cpu",
        training_source_sha256=TRAINING_SOURCE_SHA256,
    )

    manifest = result.manifest
    assert manifest["schema_version"] == 3
    assert manifest["representation_protocol_sha256"] == "a" * 64
    assert manifest["training_protocol_sha256"] == training.sha256
    assert manifest["training_unit"] == "globally_shuffled_window"
    assert manifest["batching"]["steps_per_epoch"] == 5
    assert manifest["optimizer_steps"] == 5
    assert manifest["scheduler_steps"] == 5
    assert clipping_calls == [1.0] * 5
    assert manifest["optimizer"]["scheduler"]["name"] == "OneCycleLR"
    assert manifest["optimizer"]["scheduler"]["total_steps"] == 5
    assert manifest["validation_history"][0]["learning_rate"] > 0


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
        training_source_sha256=TRAINING_SOURCE_SHA256,
    )

    assert first.reused is False
    manifest = first.manifest
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == 3
    assert manifest["representation_protocol_sha256"] == "a" * 64
    assert manifest["training_protocol_sha256"] == "2" * 64
    assert manifest["training_source_sha256"] == TRAINING_SOURCE_SHA256
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
    assert manifest["optimizer"]["scheduler"]["name"] == "OneCycleLR"
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
        training_source_sha256=TRAINING_SOURCE_SHA256,
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
            training_source_sha256=TRAINING_SOURCE_SHA256,
        )


def test_store_backed_training_is_numerically_identical_to_disk_reference(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    records = _tiny_training_records(cache_root)
    training = replace(_tiny_training_contract(), max_epochs=3, patience=3)
    disk = train_frozen_probe_run(
        head_name="mean_linear",
        seed=33,
        records=records,
        cache_root=cache_root,
        run_dir=tmp_path / "disk",
        training=training,
        device="cpu",
        training_source_sha256=TRAINING_SOURCE_SHA256,
    )
    store = ValidatedRepresentationStore.build(
        records=records,
        cache_root=cache_root,
        available_memory_bytes=16 * 1024**3,
    )
    memory = train_frozen_probe_run(
        head_name="mean_linear",
        seed=33,
        records=records,
        cache_root=cache_root,
        run_dir=tmp_path / "memory",
        training=training,
        device="cpu",
        training_source_sha256=TRAINING_SOURCE_SHA256,
        representation_store=store,
    )

    assert memory.manifest["validation_history"] == disk.manifest["validation_history"]
    assert memory.manifest["selected_epoch"] == disk.manifest["selected_epoch"]
    assert (
        memory.manifest["selected_validation_pearson"]
        == disk.manifest["selected_validation_pearson"]
    )
    disk_state = torch.load(
        tmp_path / "disk/head_checkpoint.pt", map_location="cpu", weights_only=True
    )["state_dict"]
    memory_state = torch.load(
        tmp_path / "memory/head_checkpoint.pt", map_location="cpu", weights_only=True
    )["state_dict"]
    assert all(torch.equal(disk_state[key], memory_state[key]) for key in disk_state)


def test_training_source_change_blocks_exact_resume(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    records = _tiny_training_records(cache_root)
    run_dir = tmp_path / "run"
    train_frozen_probe_run(
        head_name="mean_linear",
        seed=33,
        records=records,
        cache_root=cache_root,
        run_dir=run_dir,
        training=replace(_tiny_training_contract(), max_epochs=1, patience=1),
        device="cpu",
        training_source_sha256=TRAINING_SOURCE_SHA256,
    )

    with pytest.raises(FrozenEncoderError, match="identity does not match"):
        train_frozen_probe_run(
            head_name="mean_linear",
            seed=33,
            records=records,
            cache_root=cache_root,
            run_dir=run_dir,
            training=replace(_tiny_training_contract(), max_epochs=1, patience=1),
            device="cpu",
            training_source_sha256="2" * 64,
        )


def test_epoch_progress_event_is_complete_and_json_safe(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    records = _tiny_training_records(cache_root)
    store = ValidatedRepresentationStore.build(
        records=records,
        cache_root=cache_root,
        available_memory_bytes=16 * 1024**3,
    )
    events: list[dict[str, object]] = []
    train_frozen_probe_run(
        head_name="mean_linear",
        seed=33,
        records=records,
        cache_root=cache_root,
        run_dir=tmp_path / "run",
        training=replace(_tiny_training_contract(), max_epochs=1, patience=1),
        device="cpu",
        training_source_sha256=TRAINING_SOURCE_SHA256,
        representation_store=store,
        progress_sink=lambda event: events.append(dict(event)),
    )

    epoch = next(event for event in events if event["event"] == "epoch_complete")
    assert set(epoch) == {
        "event",
        "head_name",
        "seed",
        "epoch",
        "max_epochs",
        "training_window_mse",
        "validation_subject_mse",
        "validation_subject_pearson",
        "best_epoch",
        "best_validation_pearson",
        "epochs_without_improvement",
        "patience",
        "learning_rate",
        "optimizer_steps",
        "scheduler_steps",
        "elapsed_seconds",
    }
    assert "test" not in json.dumps(epoch)
    json.dumps(epoch, allow_nan=False)


def test_study_preflight_rejects_late_stale_run_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    records = _tiny_training_records(cache_root)
    output_root = tmp_path / "runs"
    training = replace(_tiny_training_contract(), max_epochs=1, patience=1)
    late_run = output_root / "mean_linear/seed-34"
    train_frozen_probe_run(
        head_name="mean_linear",
        seed=34,
        records=records,
        cache_root=cache_root,
        run_dir=late_run,
        training=training,
        device="cpu",
        training_source_sha256="2" * 64,
    )
    before = {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        torch.optim,
        "AdamW",
        lambda *args, **kwargs: pytest.fail("optimizer created before preflight"),
    )

    with pytest.raises(FrozenEncoderError, match="identity does not match"):
        train_frozen_probe_study(
            records=records,
            cache_root=cache_root,
            output_root=output_root,
            training=training,
            device="cpu",
            training_source_sha256=TRAINING_SOURCE_SHA256,
            available_memory_bytes=16 * 1024**3,
        )
    after = {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert after == before


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
            training_source_sha256=TRAINING_SOURCE_SHA256,
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
        training_source_sha256=TRAINING_SOURCE_SHA256,
        available_memory_bytes=16 * 1024**3,
    )

    assert inventory["status"] == "complete"
    assert inventory["schema_version"] == 3
    assert inventory["representation_protocol_sha256"] == "a" * 64
    assert inventory["training_protocol_sha256"] == training.sha256
    assert inventory["training_source_sha256"] == TRAINING_SOURCE_SHA256
    assert inventory["run_count"] == 40
    assert len(inventory["runs"]) == 40
    assert {
        (run["head_name"], run["seed"]) for run in inventory["runs"]
    } == {
        (head_name, seed)
        for head_name in APPROVED_HEADS
        for seed in range(33, 43)
    }
    assert all(
        run["representation_protocol_sha256"] == "a" * 64
        and run["training_protocol_sha256"] == training.sha256
        for run in inventory["runs"]
    )
    assert audit_frozen_probe_inventory(
        output_root=output_root,
        records=records,
        training=training,
        training_source_sha256=TRAINING_SOURCE_SHA256,
    ) == inventory

    extra = output_root / "unexpected-head"
    extra.mkdir()
    with pytest.raises(FrozenEncoderError, match="exact 40-run inventory"):
        audit_frozen_probe_inventory(
            output_root=output_root,
            records=records,
            training=training,
            training_source_sha256=TRAINING_SOURCE_SHA256,
        )


def test_study_loads_each_of_900_payloads_once_across_40_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = tuple(
        CachedSubjectRecord(
            subject_id=f"sub-{index:04d}",
            split="train" if index < 800 else "validation",
            age=float(index + 1),
            cache_identity=_identity(f"sub-{index:04d}"),
        )
        for index in range(900)
    )
    metadata = {
        -2: CachedTensorMetadata((1, 1, 1), torch.float32, 4),
        -1: CachedTensorMetadata((1, 1, 1), torch.float32, 4),
    }
    load_calls: list[str] = []
    run_calls: list[tuple[str, int]] = []

    def strict_loader(cache_root, identity, *, required_layers):
        load_calls.append(identity.subject_id)
        return {-2: torch.zeros(1, 1, 1), -1: torch.zeros(1, 1, 1)}

    def fake_train(**kwargs):
        run_calls.append((kwargs["head_name"], kwargs["seed"]))
        assert kwargs["representation_store"].subject_count == 900
        return FrozenProbeRunResult(manifest={}, reused=False)

    monkeypatch.setattr(training_module, "train_frozen_probe_run", fake_train)
    monkeypatch.setattr(
        training_module,
        "audit_frozen_probe_inventory",
        lambda **kwargs: {
            "status": "complete",
            "run_count": 40,
            "checkpoint_inventory_sha256": "f" * 64,
        },
    )

    result = train_frozen_probe_study(
        records=records,
        cache_root=tmp_path / "unused-cache",
        output_root=(tmp_path / "runs").resolve(),
        training=replace(_tiny_training_contract(), max_epochs=1, patience=1),
        device="cpu",
        training_source_sha256=TRAINING_SOURCE_SHA256,
        available_memory_bytes=16 * 1024**3,
        store_metadata_inspector=lambda *args, **kwargs: metadata,
        store_strict_loader=strict_loader,
    )

    assert result["run_count"] == 40
    assert load_calls == [record.subject_id for record in records]
    assert len(run_calls) == 40
