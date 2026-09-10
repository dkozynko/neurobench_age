from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurobench_age.pipelines.capacity_data_regime import (
    CapacityDataRegimePipelineError,
    build_capacity_data_regime_head,
    build_capacity_data_regime_summary_head,
    capacity_summary_from_tokens,
    capacity_summary_mode,
    compute_capacity_preflight,
    build_cache_manifest_from_records,
    run_capacity_data_regime,
    run_capacity_pilot,
    validate_primary_run_reuse,
    select_capacity_records,
    capacity_free_space_requirement,
    compute_capacity_preflight,
    extension_run_matrix,
    validate_extension_output_root,
)
from neurobench_age.pipelines.frozen_probe_training import FrozenProbeRunResult
from neurobench_age.pipelines.frozen_probe_training import CachedSubjectRecord
from neurobench_age.pipelines.frozen_probe import (
    RepresentationCacheIdentity,
    write_cached_representations,
)
from neurobench_age.research.capacity_data_regime_lock import build_lock_core
from neurobench_age.research.capacity_data_regime import load_capacity_data_regime_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_capacity_data_regime_protocol(
    ROOT / "configs/research/capacity_data_regime.json", repository_root=ROOT
)


def test_extension_run_matrix_is_exactly_90_identities() -> None:
    matrix = extension_run_matrix(PROTOCOL)

    assert len(matrix) == 90
    assert len({(item.training_size, item.head, item.seed) for item in matrix}) == 90
    assert {item.training_size for item in matrix} == {200, 400, 800}
    assert {item.head for item in matrix} == set(PROTOCOL.head_names)
    assert {item.seed for item in matrix} == set(range(33, 43))


def test_capacity_summary_modes_match_declared_head_families() -> None:
    assert capacity_summary_mode("mean_linear") == "mean"
    assert capacity_summary_mode("mean_mlp_residual_matched(hidden_dim=4)") == "mean"
    assert capacity_summary_mode("mean_rich_stats_residual") == "rich_stats"


@pytest.mark.parametrize(
    "head_name",
    [
        "mean_linear",
        "mean_rich_stats_residual",
        "mean_mlp_residual_matched(hidden_dim=4)",
    ],
)
def test_capacity_summary_heads_match_raw_head_outputs(head_name: str) -> None:
    tokens = torch.randn(4, 3, 2)
    torch.manual_seed(33)
    raw_head = build_capacity_data_regime_head(head_name, embed_dim=2)
    torch.manual_seed(33)
    summary_head = build_capacity_data_regime_summary_head(head_name, embed_dim=2)
    summary = capacity_summary_from_tokens(
        tokens,
        head_name=head_name,
        embed_dim=2,
    )
    with torch.inference_mode():
        assert torch.allclose(raw_head(tokens), summary_head(summary), atol=1e-6, rtol=1e-6)


def test_free_space_rule_records_measured_preflight_values() -> None:
    cache_bytes = 400
    estimated_output_bytes = 50
    expected_required = max(50 * 1024**3, 100) + 2 * estimated_output_bytes

    assert (
        capacity_free_space_requirement(
            representation_cache_bytes=cache_bytes,
            estimated_extension_output_bytes=estimated_output_bytes,
        )
        == expected_required
    )
    report = compute_capacity_preflight(
        representation_cache_bytes=cache_bytes,
        estimated_extension_output_bytes=estimated_output_bytes,
        free_space_bytes=expected_required + 1,
        available_memory_bytes=10_000,
        projected_store_bytes=100,
        train_windows=129,
        pilot_seconds=1.5,
        cached_window_count=80,
    )
    assert report == {
        "representation_cache_bytes": cache_bytes,
        "estimated_extension_output_bytes": estimated_output_bytes,
        "free_space_bytes": expected_required + 1,
        "required_free_space_bytes": expected_required,
        "peak_ram_bytes": 100,
        "upper_bound_optimizer_steps": 40 * 3,
        "observed_pilot_seconds": 1.5,
        "cached_window_count": 80,
    }


def test_free_space_rule_accepts_protocol_resource_parameters() -> None:
    expected_required = max(12 * 1024**3, 400) + 3 * 50

    assert capacity_free_space_requirement(
        representation_cache_bytes=400,
        estimated_extension_output_bytes=50,
        free_space_floor_bytes=12 * 1024**3,
        free_space_cache_fraction=0.25,
        free_space_output_multiplier=3,
    ) == expected_required


def test_preflight_rejects_insufficient_disk_or_memory() -> None:
    with pytest.raises(CapacityDataRegimePipelineError, match="free space"):
        compute_capacity_preflight(
            representation_cache_bytes=100,
            estimated_extension_output_bytes=100,
            free_space_bytes=0,
            available_memory_bytes=10_000,
            projected_store_bytes=100,
            train_windows=64,
            pilot_seconds=1.0,
            cached_window_count=1,
        )
    with pytest.raises(CapacityDataRegimePipelineError, match="memory"):
        compute_capacity_preflight(
            representation_cache_bytes=100,
            estimated_extension_output_bytes=100,
            free_space_bytes=10**15,
            available_memory_bytes=99,
            projected_store_bytes=100,
            train_windows=64,
            pilot_seconds=1.0,
            cached_window_count=1,
        )


def test_output_root_must_be_outside_repository_and_raw_data(tmp_path: Path) -> None:
    output_root = tmp_path / "extension-output"
    validate_extension_output_root(
        output_root,
        repository_root=ROOT,
        raw_data_roots=(tmp_path / "raw",),
        primary_evidence_roots=(ROOT / "results/canonical/prospective",),
    )

    with pytest.raises(CapacityDataRegimePipelineError, match="repository"):
        validate_extension_output_root(
            ROOT / "extension-output",
            repository_root=ROOT,
            raw_data_roots=(),
            primary_evidence_roots=(),
        )
    with pytest.raises(CapacityDataRegimePipelineError, match="raw"):
        validate_extension_output_root(
            tmp_path / "raw" / "extension-output",
            repository_root=ROOT,
            raw_data_roots=(tmp_path / "raw",),
            primary_evidence_roots=(),
        )


def test_extension_protocol_rejects_another_head_or_size_before_run() -> None:
    matrix = extension_run_matrix(PROTOCOL)
    assert all(item.head in PROTOCOL.head_names for item in matrix)
    assert all(item.training_size in PROTOCOL.training_sizes for item in matrix)


def test_primary_800_reuse_requires_exact_identity_fields() -> None:
    expected = {
        "head_name": "mean_linear",
        "seed": 33,
        "run_identity_sha256": "a" * 64,
        "training_protocol_sha256": "b" * 64,
        "training_source_sha256": "c" * 64,
        "environment_sha256": "d" * 64,
        "checkpoint_selection": {
            "metric": "validation_pearson",
            "mode": "max",
            "tie_break": "earliest_epoch",
        },
        "cache_contract": {"dataset_manifest_sha256": "e" * 64},
    }
    manifest = {
        "status": "complete",
        **expected,
    }

    assert validate_primary_run_reuse(manifest, expected_fields=expected) is True
    for field in expected:
        mutated = dict(manifest)
        mutated[field] = "changed" if field != "seed" else 34
        with pytest.raises(CapacityDataRegimePipelineError, match="exact primary reuse"):
            validate_primary_run_reuse(mutated, expected_fields=expected)


def test_capacity_record_selection_preserves_canonical_train_order_and_fixed_validation() -> None:
    def identity(subject_id: str) -> RepresentationCacheIdentity:
        return RepresentationCacheIdentity(
            protocol_sha256="a" * 64,
            checkpoint="brain-bzh/reve-base",
            checkpoint_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            preprocessing_sha256="d" * 64,
            subject_id=subject_id,
            source_tree_sha256="e" * 64,
        )

    records = (
        CachedSubjectRecord("a", "train", 10.0, identity("a")),
        CachedSubjectRecord("validation", "validation", 12.0, identity("validation")),
        CachedSubjectRecord("b", "train", 11.0, identity("b")),
    )
    selected = select_capacity_records(records, ("b", "a"))

    assert [record.subject_id for record in selected] == ["b", "a", "validation"]
    assert [record.split for record in selected] == ["train", "train", "validation"]


def test_runner_seals_exact_90_runs_with_injected_cpu_run_callable(tmp_path: Path) -> None:
    def identity(subject_id: str) -> RepresentationCacheIdentity:
        return RepresentationCacheIdentity(
            protocol_sha256="a" * 64,
            checkpoint="brain-bzh/reve-base",
            checkpoint_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            preprocessing_sha256="d" * 64,
            subject_id=subject_id,
            source_tree_sha256="e" * 64,
        )

    records = tuple(
        [CachedSubjectRecord(f"sub-{index:03d}", "train", 10.0, identity(f"sub-{index:03d}")) for index in range(800)]
        + [CachedSubjectRecord("validation", "validation", 12.0, identity("validation"))]
    )
    cohorts = {
        200: tuple(record.subject_id for record in records[:200]),
        400: tuple(record.subject_id for record in records[:400]),
        800: tuple(record.subject_id for record in records[:800]),
    }
    class FakeStore:
        subject_count = len(records)

        def tensor(self, subject_id: str, layer: int) -> torch.Tensor:
            assert layer == -1
            assert subject_id in {record.subject_id for record in records}
            return torch.ones(1, 1, 1)

    preflight = compute_capacity_preflight(
        representation_cache_bytes=1,
        estimated_extension_output_bytes=1,
        free_space_bytes=50 * 1024**3 + 2,
        available_memory_bytes=10_000,
        projected_store_bytes=1,
        train_windows=len(records),
        pilot_seconds=0.1,
        cached_window_count=len(records),
    )
    core = {
        "extension_id": PROTOCOL.extension_id,
        "schema_version": 1,
        "parent_primary_study_lock_sha256": "a" * 64,
        "parent_primary_prediction_inventory_sha256": "b" * 64,
        "extension_protocol_sha256": PROTOCOL.sha256,
        "representation_protocol_sha256": "c" * 64,
        "training_protocol_sha256": "d" * 64,
        "training_source_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
        "hardware_sha256": "0" * 64,
        "hbn_manifest_sha256": "1" * 64,
        "hbn_training_manifest_sha256": "2" * 64,
        "representation_cache_manifest_sha256": "3" * 64,
        "validation_subject_list_sha256": "4" * 64,
        "cohort_hashes": {"train_200": "5" * 64, "train_400": "6" * 64, "train_800": "7" * 64},
        "expected_run_count": 90,
        "expected_prediction_count": 6750,
        "output_root_identity": "test-extension-output",
        "preflight": preflight,
    }
    core_bundle = build_lock_core(core)
    primary_lock = tmp_path / "primary-study-lock.json"
    primary_inventory = tmp_path / "primary-prediction-inventory.json"
    primary_lock.write_text("lock\n", encoding="utf-8")
    primary_inventory.write_text("inventory\n", encoding="utf-8")

    def fake_run(**kwargs):
        context = kwargs["run_metadata"]
        return FrozenProbeRunResult(
                manifest={
                    "status": "complete",
                    "head_name": kwargs["head_name"],
                    "seed": kwargs["seed"],
                    "run_context": context,
                    "batching": {"total_training_windows": 1},
                    "optimizer_steps": 1,
                    "observed_early_stopping_steps": 1,
                    "validation_history": [{"epoch": 1, "validation_subject_pearson": 0.1}],
                    "selected_validation_subject_metrics": [
                        {"subject_id": "validation", "age": 12.0, "prediction": 11.0}
                    ],
                    "selected_epoch": 1,
                    "run_manifest_sha256": "8" * 64,
                    "checkpoint_sha256": "9" * 64,
                },
            reused=False,
        )

    result = run_capacity_data_regime(
        protocol=PROTOCOL,
        training=__import__(
            "neurobench_age.research.training_protocol",
            fromlist=["load_frozen_probe_training_protocol"],
        ).load_frozen_probe_training_protocol(
            ROOT / "configs/research/neuralbench_frozen_probe_training.json"
        ),
        records=records,
        cohorts=cohorts,
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "extension-output",
        repository_root=ROOT,
        raw_data_roots=(),
        primary_evidence_roots=(),
        primary_study_lock=primary_lock,
        primary_prediction_inventory=primary_inventory,
        training_source_sha256="e" * 64,
        core_bundle=core_bundle,
        preflight_report=preflight,
        device="cpu",
        representation_store=FakeStore(),
        run_callable=fake_run,
    )

    assert result["status"] == "checkpoint_sealed"
    assert result["run_count"] == 90
    assert (tmp_path / "extension-output/checkpoint_inventory.json").is_file()
    assert (tmp_path / "extension-output/checkpoint_sealed_lock.json").is_file()


def test_cache_manifest_from_records_binds_payloads_and_window_counts(tmp_path: Path) -> None:
    def identity(subject_id: str) -> RepresentationCacheIdentity:
        return RepresentationCacheIdentity(
            protocol_sha256="a" * 64,
            checkpoint="brain-bzh/reve-base",
            checkpoint_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            preprocessing_sha256="d" * 64,
            subject_id=subject_id,
            source_tree_sha256="e" * 64,
        )

    def evidence(recording: str) -> dict[str, object]:
        return {
            "encoder_frozen": True,
            "encoder_eval_mode": True,
            "inference_mode": True,
            "layer_indices": [-2, -1],
            "state_sha256_before": "f" * 64,
            "state_sha256_after": "f" * 64,
            "hbn_qc": {
                "recording_relpath": recording,
                "window_count": 2,
            },
        }

    records = (
        CachedSubjectRecord("train", "train", 10.0, identity("train")),
        CachedSubjectRecord("validation", "validation", 12.0, identity("validation")),
    )
    for record in records:
        write_cached_representations(
            tmp_path,
            record.cache_identity,
            {-2: torch.ones(2, 1, 1), -1: torch.ones(2, 1, 1)},
            evidence=evidence(f"{record.subject_id}.set"),
        )

    manifest, digest, cache_bytes, cached_window_count, train_window_count = build_cache_manifest_from_records(
        records, cache_root=tmp_path
    )

    assert len(manifest) == 4
    assert len(digest) == 64
    assert cache_bytes == 32
    assert cached_window_count == 4
    assert train_window_count == 2
    assert manifest[0]["split"] == "train"
    assert manifest[-1]["split"] == "validation"


def test_capacity_pilot_uses_a_declared_extension_identity(tmp_path: Path) -> None:
    def identity(subject_id: str) -> RepresentationCacheIdentity:
        return RepresentationCacheIdentity(
            protocol_sha256="a" * 64,
            checkpoint="brain-bzh/reve-base",
            checkpoint_sha256="b" * 64,
            dataset_manifest_sha256="c" * 64,
            preprocessing_sha256="d" * 64,
            subject_id=subject_id,
            source_tree_sha256="e" * 64,
        )

    records = tuple(
        [CachedSubjectRecord(f"sub-{index:03d}", "train", 10.0, identity(f"sub-{index:03d}")) for index in range(200)]
        + [CachedSubjectRecord("validation", "validation", 12.0, identity("validation"))]
    )

    class FakeStore:
        subject_count = len(records)

        def tensor(self, subject_id: str, layer: int) -> torch.Tensor:
            assert subject_id
            assert layer == -1
            return torch.ones(2, 1, 2)

    def fake_run(**kwargs):
        return FrozenProbeRunResult(
            manifest={
                "status": "complete",
                "head_name": kwargs["head_name"],
                "seed": kwargs["seed"],
                "run_context": kwargs["run_metadata"],
                "runtime_seconds": 1.25,
                "run_manifest_sha256": "8" * 64,
            },
            reused=False,
        )

    result = run_capacity_pilot(
        protocol=PROTOCOL,
        training=__import__(
            "neurobench_age.research.training_protocol",
            fromlist=["load_frozen_probe_training_protocol"],
        ).load_frozen_probe_training_protocol(
            ROOT / "configs/research/neuralbench_frozen_probe_training.json"
        ),
        records=records,
        cohorts={200: tuple(record.subject_id for record in records[:200])},
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "pilot",
        device="cpu",
        training_source_sha256="e" * 64,
        representation_store=FakeStore(),
        run_callable=fake_run,
    )

    assert result["status"] == "pilot_complete"
    assert result["observed_pilot_seconds"] == 1.25
