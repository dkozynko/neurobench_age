from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
import neurobench_age.research.sealing as sealing_module

from neurobench_age.research.study_lock import (
    StudyLockError,
    fail_study,
    load_study_lock,
    record_resumable_failure,
    seal_study,
    transition_study,
    verify_exact_study,
)
from neurobench_age.pipelines.frozen_probe_training import APPROVED_HEADS
from neurobench_age.research.protocol import load_study_protocol


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_inventory(tmp_path: Path) -> tuple[Path, str]:
    body = {
        "schema_version": 3,
        "status": "complete",
        "representation_protocol_sha256": "a" * 64,
        "training_protocol_sha256": "9" * 64,
        "training_source_sha256": "b" * 64,
        "heads": list(APPROVED_HEADS),
        "seeds": list(range(33, 43)),
        "run_count": 40,
        "runs": [
            {
                "head_name": head_name,
                "seed": seed,
                "representation_protocol_sha256": "a" * 64,
                "training_protocol_sha256": "9" * 64,
                "training_source_sha256": "b" * 64,
                "run_identity_sha256": "1" * 64,
                "run_manifest_sha256": "2" * 64,
                "checkpoint_sha256": "3" * 64,
                "selected_epoch": 1,
                "head_parameter_count": 10,
            }
            for head_name in APPROVED_HEADS
            for seed in range(33, 43)
        ],
    }
    from neurobench_age.research.study_lock import canonical_sha256

    digest = canonical_sha256(body)
    path = tmp_path / "checkpoint_inventory.json"
    path.write_text(
        json.dumps({**body, "checkpoint_inventory_sha256": digest}) + "\n"
    )
    return path, digest


def _payload(tmp_path: Path) -> dict[str, object]:
    digest = "a" * 64
    _, checkpoint_inventory_sha256 = _checkpoint_inventory(tmp_path)
    return {
        "study_id": "reve_age_external_frozen_probe_v1",
        "protocol_sha256": digest,
        "training_protocol_sha256": "9" * 64,
        "representation_source_sha256": "8" * 64,
        "training_source_sha256": "b" * 64,
        "git_revision": "revision",
        "git_dirty": False,
        "encoder_checkpoint": "brain-bzh/reve-base",
        "encoder_checkpoint_sha256": "c" * 64,
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "environment_sha256": "4" * 64,
        "hbn_manifest_sha256": "d" * 64,
        "hbn_training_manifest_sha256": "5" * 64,
        "mipdb_manifest_sha256": "e" * 64,
        "mipdb_pilot_qc_sha256": "6" * 64,
        "mipdb_cohort_qc_sha256": "7" * 64,
        "subject_list_sha256": {
            "hbn_train": digest,
            "hbn_validation": digest,
            "mipdb_pilot": digest,
            "mipdb_primary": digest,
            "mipdb_extrapolation": digest,
        },
        "heads": [
            "mean_linear",
            "mean_layer_linear",
            "mean_rich_stats_residual",
            "multi_query_rich_stats",
        ],
        "seeds": list(range(33, 43)),
        "preprocessing_sha256": "f" * 64,
        "statistics_sha256": "1" * 64,
        "output_root": str(tmp_path / "external-output"),
    }


def test_seal_creates_immutable_lock_and_separate_state(tmp_path: Path) -> None:
    lock_path = tmp_path / "study_lock.json"

    lock = seal_study(lock_path, _payload(tmp_path))

    assert lock["lock_sha256"] == load_study_lock(lock_path)["lock_sha256"]
    state = json.loads((tmp_path / "study_state.json").read_text())
    assert state["state"] == "sealed"
    with pytest.raises(StudyLockError, match="already exists"):
        seal_study(lock_path, _payload(tmp_path))


def test_seal_requires_pilot_and_final_cohort_qc_artifacts(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload.pop("mipdb_pilot_qc_sha256")

    with pytest.raises(StudyLockError, match="mipdb_pilot_qc_sha256") as error:
        seal_study(tmp_path / "study_lock.json", payload)
    assert "missing=['mipdb_pilot_qc_sha256']" in str(error.value)


def test_checkpoint_artifact_audit_rejects_self_hashed_inventory_without_files(
    tmp_path: Path,
) -> None:
    inventory_path, _ = _checkpoint_inventory(tmp_path)

    with pytest.raises(StudyLockError, match="checkpoint.*(missing|not exact)"):
        sealing_module.verify_checkpoint_artifacts(
            checkpoint_root=tmp_path / "runs",
            inventory_path=inventory_path,
        )


def test_checkpoint_inventory_carries_both_protocol_identities(tmp_path: Path) -> None:
    inventory_path, _ = _checkpoint_inventory(tmp_path)

    from neurobench_age.research.study_lock import load_checkpoint_inventory

    inventory = load_checkpoint_inventory(inventory_path)

    assert inventory["schema_version"] == 3
    assert inventory["representation_protocol_sha256"] == "a" * 64
    assert inventory["training_protocol_sha256"] == "9" * 64
    assert all(
        run["representation_protocol_sha256"] == "a" * 64
        and run["training_protocol_sha256"] == "9" * 64
        for run in inventory["runs"]
    )


def test_production_sealing_cli_is_artifact_derived() -> None:
    source = (ROOT / "scripts/seal_external_study.py").read_text(encoding="utf-8")

    assert 'add_argument("--payload"' not in source
    for required in (
        "--environment",
        "--hbn-subject-manifest",
        "--hbn-training-manifest",
        "--hbn-data-root",
        "--checkpoint-root",
        "--mipdb-manifest",
        "--mipdb-bids-root",
        "--mipdb-pilot-qc",
        "--mipdb-cohort-qc",
        "--output-root",
    ):
        assert f'add_argument("{required}"' in source


def test_lifecycle_allows_only_declared_transitions(tmp_path: Path) -> None:
    lock_path = tmp_path / "study_lock.json"
    seal_study(lock_path, _payload(tmp_path))

    assert transition_study(lock_path, "started")["state"] == "started"
    assert transition_study(lock_path, "completed")["state"] == "completed"
    with pytest.raises(StudyLockError, match="terminal"):
        transition_study(lock_path, "started")


def test_lifecycle_rejects_skipping_started_state(tmp_path: Path) -> None:
    lock_path = tmp_path / "study_lock.json"
    seal_study(lock_path, _payload(tmp_path))

    with pytest.raises(StudyLockError, match="sealed -> completed"):
        transition_study(lock_path, "completed")


def test_lock_detects_tampering_and_expected_provenance_drift(tmp_path: Path) -> None:
    lock_path = tmp_path / "study_lock.json"
    payload = _payload(tmp_path)
    seal_study(lock_path, payload)

    changed = dict(payload)
    changed["mipdb_manifest_sha256"] = "9" * 64
    changed["training_source_sha256"] = "8" * 64
    with pytest.raises(StudyLockError) as raised:
        verify_exact_study(lock_path, changed)
    assert raised.value.mismatched_fields == (
        "mipdb_manifest_sha256",
        "training_source_sha256",
    )

    tampered = json.loads(lock_path.read_text())
    tampered["encoder_checkpoint"] = "other/model"
    lock_path.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(StudyLockError, match="digest"):
        load_study_lock(lock_path)


def test_failure_is_diagnostic_and_does_not_rewrite_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "study_lock.json"
    seal_study(lock_path, _payload(tmp_path))
    transition_study(lock_path, "started")
    before = lock_path.read_bytes()

    state = fail_study(lock_path, error="synthetic inference failure")

    assert state["state"] == "failed"
    assert lock_path.read_bytes() == before
    failure = json.loads((tmp_path / "study_failure.json").read_text())
    assert failure["error"] == "synthetic inference failure"
    with pytest.raises(StudyLockError, match="terminal"):
        transition_study(lock_path, "completed")


def test_resumable_failure_is_append_only_and_keeps_started_state(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "study_lock.json"
    seal_study(lock_path, _payload(tmp_path))
    transition_study(lock_path, "started")

    first = record_resumable_failure(lock_path, error="temporary GPU failure")
    second = record_resumable_failure(lock_path, error="temporary disk failure")

    assert first["attempt"] == 1
    assert second["attempt"] == 2
    assert json.loads((tmp_path / "study_state.json").read_text())["state"] == "started"
    evidence = sorted((tmp_path / "study_failures").glob("*.json"))
    assert len(evidence) == 2
    assert json.loads(evidence[0].read_text())["error"] == "temporary GPU failure"


def test_sealing_cli_loads_the_same_protocol_and_records_its_digest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    protocol_path = ROOT / "configs" / "research" / "external_frozen_probe.json"
    protocol = load_study_protocol(protocol_path)
    payload = _payload(tmp_path)
    payload["protocol_sha256"] = protocol.sha256
    payload["statistics_sha256"] = protocol.statistics_sha256
    lock_path = tmp_path / "study_lock.json"
    script_path = ROOT / "scripts/seal_external_study.py"
    spec = importlib.util.spec_from_file_location("seal_external_study", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "derive_study_payload", lambda **kwargs: payload)
    paths = {
        name: (tmp_path / name).resolve()
        for name in (
            "environment",
            "hbn-subject-manifest",
            "hbn-training-manifest",
            "hbn-data-root",
            "checkpoint-root",
            "checkpoint-inventory",
            "mipdb-manifest",
            "mipdb-bids-root",
            "mipdb-pilot-qc",
            "mipdb-cohort-qc",
        )
    }

    assert module.main(
        [
            "--protocol",
            str(protocol_path),
            *[
                value
                for name, path in paths.items()
                for value in (f"--{name}", str(path))
            ],
            "--output-root",
            str((tmp_path / "external-output").resolve()),
            "--lock",
            str(lock_path.resolve()),
        ]
    ) == 0

    capsys.readouterr()
    assert load_study_lock(lock_path)["protocol_sha256"] == load_study_protocol(
        protocol_path
    ).sha256
    assert load_study_lock(lock_path)["statistics_sha256"] == protocol.statistics_sha256


def test_checkpoint_inventory_loader_rejects_incomplete_inventory(tmp_path: Path) -> None:
    from neurobench_age.research.study_lock import load_checkpoint_inventory

    inventory_path, _ = _checkpoint_inventory(tmp_path)
    inventory = json.loads(inventory_path.read_text())
    inventory["run_count"] = 39
    inventory_path.write_text(json.dumps(inventory) + "\n")

    with pytest.raises(StudyLockError, match="digest"):
        load_checkpoint_inventory(inventory_path)
