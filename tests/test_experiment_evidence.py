from __future__ import annotations

import json
from pathlib import Path

import pytest

import neurobench_age.core.evidence as experiment_evidence
from neurobench_age.core.evidence import (
    EvidenceRecorder,
    add_declared_head_bucket,
    estimate_head_parameter_count,
    validate_parameter_buckets,
)


def _recorder(tmp_path: Path, *, evaluation_mode: str = "validation_only") -> EvidenceRecorder:
    return EvidenceRecorder(
        tmp_path / "run",
        run_id="run-33",
        task="age",
        dataset_manifest="manifest-sha",
        split_fingerprint="split-sha",
        seed=33,
        resolved_config={"head_variant": "mean_linear"},
        command_line="python run.py --head-variant mean_linear",
        evaluation_mode=evaluation_mode,
    )


def test_recorder_writes_schema_versioned_running_manifest_and_finalizes(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)

    manifest = recorder.start()

    assert manifest["schema_version"] == "1.0"
    assert manifest["status"] == "running"
    assert manifest["run_id"] == "run-33"
    assert manifest["evaluation_mode"] == "validation_only"
    assert manifest["test_access"] == "sealed"
    assert manifest["config_hash"]
    assert (tmp_path / "run" / "run_manifest.json").is_file()

    recorder.add_missing("cpu_rss unavailable")
    finalized = recorder.finalize("completed")

    assert finalized["status"] == "completed"
    assert finalized["ended_at_utc"]
    persisted = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert persisted["missing"] == [
        "cpu_rss unavailable",
        "GPU peak memory unavailable (CUDA not available)",
    ]
    assert persisted["source_tree_sha256"]
    assert persisted["protocol_digest"]
    assert persisted["comparison_config_hash"]
    assert persisted["deterministic_policy"] == "best_effort"


@pytest.mark.parametrize(
    ("status", "error"),
    [("failed", "training exploded"), ("partial", "external interruption")],
)
def test_recorder_persists_failure_or_partial_reason(
    tmp_path: Path, status: str, error: str
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()

    finalized = recorder.finalize(status, error=error)

    assert finalized["status"] == status
    assert finalized["failure_reason"] == error
    persisted = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert persisted["failure_reason"] == error


def test_parameter_buckets_include_auxiliary_modules_and_validate_totals() -> None:
    buckets = {
        "encoder": {"total": 100, "trainable": 80, "frozen": 20},
        "head": {"total": 10, "trainable": 10, "frozen": 0},
        "auxiliary": [
            {"name": "adapter", "total": 5, "trainable": 5, "frozen": 0}
        ],
        "total": {"total": 115, "trainable": 95, "frozen": 20},
    }

    validate_parameter_buckets(buckets)


def test_parameter_bucket_validation_rejects_inconsistent_totals() -> None:
    buckets = {
        "encoder": {"total": 100, "trainable": 80, "frozen": 20},
        "head": {"total": 10, "trainable": 10, "frozen": 0},
        "auxiliary": [],
        "total": {"total": 111, "trainable": 90, "frozen": 20},
    }

    with pytest.raises(ValueError, match="parameter total"):
        validate_parameter_buckets(buckets)


def test_declared_head_bucket_completes_external_head_accounting() -> None:
    buckets = {
        "encoder": {"total": 100, "trainable": 100, "frozen": 0},
        "head": {"total": 0, "trainable": 0, "frozen": 0},
        "auxiliary": [],
        "total": {"total": 100, "trainable": 100, "frozen": 0},
    }

    completed = add_declared_head_bucket(buckets, parameter_count=513)

    assert completed["head"] == {"total": 513, "trainable": 513, "frozen": 0}
    assert completed["total"] == {"total": 613, "trainable": 613, "frozen": 0}
    validate_parameter_buckets(completed)


def test_source_tree_digest_is_stable_and_changes_with_source_content(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "runner.py"
    source_file.write_text("print('one')\n")

    first = experiment_evidence.source_tree_sha256(source_root)
    second = experiment_evidence.source_tree_sha256(source_root)
    source_file.write_text("print('two')\n")

    assert first == second
    assert first != experiment_evidence.source_tree_sha256(source_root)


def test_source_tree_digest_ignores_runtime_artifacts(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "runner.py").write_text("print('one')\n")
    first = experiment_evidence.source_tree_sha256(source_root)
    (source_root / "results").mkdir()
    (source_root / "results" / "run_manifest.json").write_text("{}\n")
    (source_root / "__pycache__").mkdir()
    (source_root / "__pycache__" / "runner.pyc").write_bytes(b"cache")

    assert experiment_evidence.source_tree_sha256(source_root) == first


def test_git_metadata_hashes_repository_root_after_src_migration() -> None:
    metadata = experiment_evidence._git_metadata()
    repository_root = Path(__file__).resolve().parents[1]

    assert metadata["source_tree_sha256"] == experiment_evidence.source_tree_sha256(repository_root)


def test_comparison_config_hash_excludes_declared_factor_but_keeps_protocol(tmp_path: Path) -> None:
    base = {
        "protocol": {"monitor": "val/pearsonr", "epochs": 3},
        "head_variant": "mean_linear",
        "head_complexity": {"parameter_count": 513},
        "learning_rate": 1e-4,
        "output_dir": str(tmp_path / "baseline"),
    }
    candidate = {
        **base,
        "head_variant": "mean_rich_stats_residual",
        "head_complexity": {"parameter_count": 2561},
        "output_dir": str(tmp_path / "candidate"),
    }
    changed_protocol = {**candidate, "protocol": {"monitor": "test/pearsonr", "epochs": 3}}

    assert experiment_evidence.comparison_config_hash(base) == experiment_evidence.comparison_config_hash(candidate)
    assert experiment_evidence.comparison_config_hash(base) != experiment_evidence.comparison_config_hash(changed_protocol)


def test_comparison_config_hash_uses_generated_neuralbench_config_fields() -> None:
    base = {
        "H7_HEAD_VARIANT": "mean_linear",
        "H7_LAYER_INDEX": -1,
        "ROBUST_LOSS": "mse",
        "DATA_DIR": "/data/one",
        "SAVE_DIR": "/results/one",
        "N_CPUS": 2,
    }
    candidate = {
        **base,
        "H7_HEAD_VARIANT": "mean_rich_stats_residual",
        "DATA_DIR": "/data/two",
        "SAVE_DIR": "/results/two",
    }
    changed_runtime = {**candidate, "N_CPUS": 4}

    assert experiment_evidence.comparison_config_hash(base) == experiment_evidence.comparison_config_hash(candidate)
    assert experiment_evidence.comparison_config_hash(base) != experiment_evidence.comparison_config_hash(changed_runtime)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("mean_linear", 513),
        ("mean_layer_mix", 514),
        ("mean_rich_stats_residual", 2561),
    ],
)
def test_estimate_head_parameter_count_for_current_heads(variant: str, expected: int) -> None:
    assert estimate_head_parameter_count(variant, embed_dim=512, n_outputs=1) == expected


class _FakeResourceProbe:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "gpu_model": "Fake GPU",
            "gpu_count": 1,
            "gpu_vram_mb": 24576,
            "hardware_class": "Fake GPU/24GB",
            "cuda": "12.8",
            "driver": "555.42",
            "peak_allocated_mb": 123.5,
            "peak_reserved_mb": 256.0,
        }


def test_phase_records_elapsed_time_throughput_and_gpu_resource_snapshot(tmp_path: Path) -> None:
    probe = _FakeResourceProbe()
    recorder = EvidenceRecorder(
        tmp_path / "run",
        run_id="run-33",
        task="age",
        dataset_manifest="manifest-sha",
        split_fingerprint="split-sha",
        seed=33,
        resolved_config={"head_variant": "mean_linear"},
        command_line="python run.py",
        resource_probe=probe,
    )
    recorder.start()

    with recorder.phase("train") as phase:
        phase.record(batches=4, samples=8)

    recorder.finalize("completed")

    complexity = json.loads((tmp_path / "run" / "complexity.json").read_text())
    assert probe.reset_calls == 1
    assert complexity["phases"]["train"]["batches"] == 4
    assert complexity["phases"]["train"]["samples"] == 8
    assert complexity["phases"]["train"]["elapsed_seconds"] >= 0
    assert complexity["phases"]["train"]["samples_per_second"] >= 0
    assert complexity["memory"]["peak_allocated_mb"] == 123.5
    assert complexity["hardware"]["hardware_class"] == "Fake GPU/24GB"


def test_cpu_resource_fallback_records_explicit_missing_reason(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start()
    recorder.finalize("completed")

    complexity = json.loads((tmp_path / "run" / "complexity.json").read_text())
    assert complexity["memory"]["peak_allocated_mb"] is None
    assert any("GPU" in reason for reason in complexity["missing"])
