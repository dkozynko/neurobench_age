from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from neurobench_age.research.study_lock import (
    StudyLockError,
    fail_study,
    load_study_lock,
    seal_study,
    transition_study,
    verify_exact_study,
)
from neurobench_age.research.protocol import load_study_protocol


ROOT = Path(__file__).resolve().parents[1]


def _payload(tmp_path: Path) -> dict[str, object]:
    digest = "a" * 64
    return {
        "study_id": "reve_age_external_frozen_probe_v1",
        "protocol_sha256": digest,
        "source_tree_sha256": "b" * 64,
        "git_revision": "revision",
        "git_dirty": False,
        "encoder_checkpoint": "brain-bzh/reve-base",
        "encoder_checkpoint_sha256": "c" * 64,
        "hbn_manifest_sha256": "d" * 64,
        "mipdb_manifest_sha256": "e" * 64,
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
    changed["source_tree_sha256"] = "8" * 64
    with pytest.raises(StudyLockError) as raised:
        verify_exact_study(lock_path, changed)
    assert raised.value.mismatched_fields == (
        "mipdb_manifest_sha256",
        "source_tree_sha256",
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


def test_sealing_cli_loads_the_same_protocol_and_records_its_digest(tmp_path: Path) -> None:
    protocol_path = ROOT / "configs" / "research" / "external_frozen_probe.json"
    payload = _payload(tmp_path)
    payload["protocol_sha256"] = load_study_protocol(protocol_path).sha256
    payload_path = tmp_path / "seal_payload.json"
    payload_path.write_text(json.dumps(payload) + "\n")
    lock_path = tmp_path / "study_lock.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "seal_external_study.py"),
            "--protocol",
            str(protocol_path),
            "--payload",
            str(payload_path),
            "--lock",
            str(lock_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert load_study_lock(lock_path)["protocol_sha256"] == load_study_protocol(
        protocol_path
    ).sha256
