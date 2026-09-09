from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.analysis import mipdb_aggregate
from neurobench_age.research.study_lock import HEADS, SUBJECT_LISTS, seal_study, transition_study


ROOT = Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _lock(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    sha = {field: _sha(field) for field in (
        "protocol_sha256",
        "training_protocol_sha256",
        "representation_source_sha256",
        "training_source_sha256",
        "encoder_checkpoint_sha256",
        "checkpoint_inventory_sha256",
        "environment_sha256",
        "hbn_manifest_sha256",
        "hbn_training_manifest_sha256",
        "mipdb_manifest_sha256",
        "mipdb_pilot_qc_sha256",
        "mipdb_cohort_qc_sha256",
        "preprocessing_sha256",
        "statistics_sha256",
    )}
    payload = {
        "study_id": "aggregate-test",
        **sha,
        "git_revision": "abc123",
        "git_dirty": False,
        "encoder_checkpoint": "brain-bzh/reve-base",
        "subject_list_sha256": {name: _sha(name) for name in SUBJECT_LISTS},
        "heads": list(HEADS),
        "seeds": list(range(33, 43)),
        "output_root": str((tmp_path / "run-output").resolve()),
    }
    path = tmp_path / "study_lock.json"
    lock = seal_study(path, payload)
    transition_study(path, "started")
    transition_study(path, "completed")
    return path, lock


def test_cli_candidate_generation_requires_pinned_source_and_writes_no_ids(tmp_path: Path) -> None:
    participants = tmp_path / "participants.tsv"
    participants.write_text(
        "participant_id\tage\nsub-01\t12\nsub-02\t22\n",
        encoding="utf-8",
    )
    draft = tmp_path / "draft.json"
    final = tmp_path / "final.json"
    qc = tmp_path / "qc.json"
    _write_json(draft, {"schema_version": 2, "status": "draft", "subjects": [], "cohorts": {}})
    _write_json(final, {"schema_version": 2, "status": "finalized", "subjects": [], "cohorts": {}})
    _write_json(qc, {"schema_version": 1, "status": "complete", "subjects": []})
    source_identity = tmp_path / "source_identity.json"
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("source", encoding="utf-8")
    _write_json(
        source_identity,
        {
            "provider": "NEMAR",
            "dataset_id": "nm000153",
            "release_id": "v1.0.0",
            "source_uri": "https://data.nemar.org/nm000153/v1.0.0/manifest.json",
            "source_manifest_sha256": _sha("source"),
            "mipdb_inventory_sha256": _sha("wrong"),
        },
    )
    source_hashes = tmp_path / "source_hashes.json"
    _write_json(source_hashes, {field: _sha(field) for field in (
        "study_lock_sha256", "protocol_sha256", "training_protocol_sha256",
        "mipdb_manifest_sha256", "cohort_qc_sha256", "preprocessing_sha256",
    )})
    study_lock, _ = _lock(tmp_path)
    output = tmp_path / "candidate.json"
    script = ROOT / "scripts" / "build_mipdb_aggregate.py"

    import importlib.util

    spec = importlib.util.spec_from_file_location("build_mipdb_aggregate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.main([
            "candidate",
            "--participants", str(participants),
            "--draft-manifest", str(draft),
            "--final-manifest", str(final),
            "--cohort-qc", str(qc),
            "--source-manifest-file", str(source_manifest),
            "--source-identity", str(source_identity),
            "--source-hashes", str(source_hashes),
            "--study-lock", str(study_lock),
            "--candidate-output", str(output),
        ])
    assert not output.exists()


def test_cli_candidate_generation_binds_all_files_and_omits_subject_ids(tmp_path: Path) -> None:
    subject_ids = [f"sub-{index:02d}" for index in range(1, 8)]
    participants = tmp_path / "participants.tsv"
    participants.write_text(
        "participant_id\tage\tsex\n"
        + "\n".join(f"{subject}\t{20 + index}\tF" for index, subject in enumerate(subject_ids))
        + "\n",
        encoding="utf-8",
    )
    subjects = [
        {
            "subject_id": subject,
            "age": 20 + index,
            "recordings": [f"{subject}/eeg/{subject}_task-block01_eeg.vhdr"],
        }
        for index, subject in enumerate(subject_ids)
    ]
    draft = tmp_path / "draft.json"
    _write_json(
        draft,
        {
            "schema_version": 2,
            "status": "draft",
            "subjects": subjects,
            "cohorts": {
                "pilot": [subject_ids[0]],
                "primary": subject_ids[1:6],
                "extrapolation": [subject_ids[6]],
            },
            "exclusions": [],
        },
    )
    dataset_identity = _sha("dataset-identity")
    final = tmp_path / "final.json"
    draft_hash = mipdb_aggregate.sha256_file(draft)
    qc_body = {
        "schema_version": 1,
        "status": "complete",
        "dataset_manifest_sha256": dataset_identity,
        "draft_manifest_sha256": draft_hash,
        "subjects": [
            {"subject_id": subject, "status": "passed", "window_count": 60}
            for subject in subject_ids[1:]
        ],
    }
    qc = tmp_path / "qc.json"
    qc_digest = mipdb_aggregate.canonical_sha256(qc_body)
    _write_json(qc, {**qc_body, "cohort_qc_sha256": qc_digest})
    final_payload = {
        "schema_version": 2,
        "status": "finalized",
        "dataset_manifest_sha256": dataset_identity,
        "draft_manifest_sha256": draft_hash,
        "cohort_qc_sha256": qc_digest,
        "subjects": subjects,
        "cohorts": {
            "pilot": [subject_ids[0]],
            "primary": subject_ids[1:6],
            "extrapolation": [subject_ids[6]],
        },
        "exclusions": [],
    }
    _write_json(final, final_payload)

    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("pinned source manifest\n", encoding="utf-8")
    final_hash = mipdb_aggregate.sha256_file(final)
    qc_hash = mipdb_aggregate.sha256_file(qc)
    lock_sha = {field: _sha(field) for field in (
        "protocol_sha256",
        "training_protocol_sha256",
        "representation_source_sha256",
        "training_source_sha256",
        "encoder_checkpoint_sha256",
        "checkpoint_inventory_sha256",
        "environment_sha256",
        "hbn_manifest_sha256",
        "hbn_training_manifest_sha256",
        "mipdb_manifest_sha256",
        "mipdb_pilot_qc_sha256",
        "mipdb_cohort_qc_sha256",
        "preprocessing_sha256",
        "statistics_sha256",
    )}
    lock_sha["mipdb_manifest_sha256"] = final_hash
    lock_sha["mipdb_cohort_qc_sha256"] = qc_hash
    lock_payload = {
        "study_id": "aggregate-cli-success",
        **lock_sha,
        "git_revision": "abc123",
        "git_dirty": False,
        "encoder_checkpoint": "brain-bzh/reve-base",
        "subject_list_sha256": {name: _sha(name) for name in SUBJECT_LISTS},
        "heads": list(HEADS),
        "seeds": list(range(33, 43)),
        "output_root": str((tmp_path / "run-output").resolve()),
    }
    study_lock = tmp_path / "study_lock.json"
    lock = seal_study(study_lock, lock_payload)
    transition_study(study_lock, "started")
    transition_study(study_lock, "completed")

    source_identity = tmp_path / "source_identity.json"
    _write_json(
        source_identity,
        {
            "provider": "NEMAR",
            "dataset_id": "nm000153",
            "release_id": "v1.0.0",
            "source_uri": "https://data.nemar.org/nm000153/v1.0.0/manifest.json",
            "source_manifest_sha256": mipdb_aggregate.sha256_file(source_manifest),
            "mipdb_inventory_sha256": final_hash,
        },
    )
    source_hashes = tmp_path / "source_hashes.json"
    _write_json(
        source_hashes,
        {
            "study_lock_sha256": lock["lock_sha256"],
            "protocol_sha256": lock["protocol_sha256"],
            "training_protocol_sha256": lock["training_protocol_sha256"],
            "mipdb_manifest_sha256": final_hash,
            "cohort_qc_sha256": qc_hash,
            "preprocessing_sha256": lock["preprocessing_sha256"],
        },
    )
    output = tmp_path / "candidate.json"
    script = ROOT / "scripts" / "build_mipdb_aggregate.py"

    import importlib.util

    spec = importlib.util.spec_from_file_location("build_mipdb_aggregate_success", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([
        "candidate",
        "--participants", str(participants),
        "--draft-manifest", str(draft),
        "--final-manifest", str(final),
        "--cohort-qc", str(qc),
        "--source-manifest-file", str(source_manifest),
        "--source-identity", str(source_identity),
        "--source-hashes", str(source_hashes),
        "--study-lock", str(study_lock),
        "--candidate-output", str(output),
    ]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = output.read_text(encoding="utf-8")
    assert all(subject not in serialized for subject in subject_ids)
    assert payload["demographics"] == {
        "sex": [{"category": "female", "count": 5, "suppressed": False}]
    }
