from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

import neurobench_age.data.mipdb as mipdb_module
from neurobench_age.data.mipdb import MipdbInventoryError, build_mipdb_inventory
from neurobench_age.research.protocol import load_study_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_study_protocol(
    ROOT / "configs" / "research" / "external_frozen_probe.json"
)


def _dataset(root: Path, rows: list[tuple[str, str]], recordings: set[str]) -> None:
    root.mkdir()
    (root / "dataset_description.json").write_text(
        '{"Name":"MIPDB synthetic","DatasetType":"raw"}\n'
    )
    (root / "participants.tsv").write_text(
        "participant_id\tage\n"
        + "".join(f"{subject}\t{age}\n" for subject, age in rows)
    )
    for subject in recordings:
        eeg = root / subject / "eeg"
        eeg.mkdir(parents=True)
        (eeg / f"{subject}_task-rest_eeg.set").write_bytes(b"metadata-only-fixture")


def _pilot_score(dataset_sha: str, subject_id: str) -> str:
    raw = (
        dataset_sha
        + "\0"
        + subject_id
        + "\0"
        + "mipdb-engineering-pilot-v1"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_inventory_is_deterministic_and_pilot_uses_declared_hash_rule(
    tmp_path: Path,
) -> None:
    rows = [(f"sub-{index:03d}", str(6 + index / 2)) for index in range(30)]
    _dataset(tmp_path / "mipdb", list(reversed(rows)), {subject for subject, _ in rows})

    first = build_mipdb_inventory(
        tmp_path / "mipdb", protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )
    second = build_mipdb_inventory(
        tmp_path / "mipdb", protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )

    assert first == second
    assert [row["subject_id"] for row in first["subjects"]] == sorted(
        subject for subject, _ in rows
    )
    expected = sorted(
        (subject for subject, _ in rows),
        key=lambda subject: (_pilot_score(first["dataset_manifest_sha256"], subject), subject),
    )[:10]
    assert first["cohorts"]["pilot"] == expected
    assert not set(first["cohorts"]["pilot"]) & set(first["cohorts"]["primary"])
    assert "metrics" not in first
    assert "predictions" not in first
    assert first["status"] == "draft"


def test_target_free_qc_finalizes_cohort_and_records_fixed_exclusions(
    tmp_path: Path,
) -> None:
    rows = [(f"sub-{index:03d}", "12") for index in range(14)]
    root = tmp_path / "mipdb"
    _dataset(root, rows, {subject for subject, _ in rows})
    draft = build_mipdb_inventory(
        root, protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft, sort_keys=True) + "\n", encoding="utf-8")
    rejected = draft["cohorts"]["primary"][0]

    def subject_loader(bids_root, subject, *, contract):
        if subject["subject_id"] == rejected:
            raise mipdb_module.MipdbPreprocessingError("synthetic bad markers")
        return np.ones((2, 128, 400), dtype=np.float32), {
            "window_count": 2,
            "qc_reasons": [],
            "mapped_channel_count": 128,
            "cross_block_windows": False,
            "spatial_interpolation": False,
        }

    qc_path = tmp_path / "cohort_qc.json"
    final_path = tmp_path / "final.json"
    finalized = mipdb_module.finalize_mipdb_cohort(
        bids_root=root,
        draft_manifest_path=draft_path,
        protocol=PROTOCOL,
        qc_output_path=qc_path,
        output_path=final_path,
        subject_loader=subject_loader,
    )

    assert finalized["status"] == "finalized"
    assert rejected not in finalized["cohorts"]["primary"]
    exclusion = next(
        item for item in finalized["exclusions"] if item["subject_id"] == rejected
    )
    assert exclusion["reason"] == "predeclared_signal_qc_failed"
    assert exclusion["detail"] == "synthetic bad markers"
    assert finalized["underpowered"] is True
    assert len(finalized["cohort_qc_sha256"]) == 64
    assert json.loads(qc_path.read_text())["status"] == "complete"
    assert json.loads(final_path.read_text()) == finalized


def test_inventory_records_model_free_exclusions_and_power_warning(tmp_path: Path) -> None:
    valid = [(f"sub-{index:03d}", "12") for index in range(12)]
    rows = valid + [("sub-missing-age", "n/a"), ("sub-bad-age", "unknown"), ("sub-no-eeg", "14")]
    recordings = {subject for subject, _ in valid} | {"sub-missing-age", "sub-bad-age"}
    _dataset(tmp_path / "mipdb", rows, recordings)

    result = build_mipdb_inventory(
        tmp_path / "mipdb", protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )

    reasons = {row["subject_id"]: row["reason"] for row in result["exclusions"]}
    assert reasons == {
        "sub-bad-age": "invalid_age",
        "sub-missing-age": "missing_age",
        "sub-no-eeg": "missing_eeg_recording",
    }
    assert result["underpowered"] is True


def test_inventory_counts_brainvision_header_not_binary_companion(tmp_path: Path) -> None:
    rows = [(f"sub-{index:03d}", "12") for index in range(12)]
    root = tmp_path / "mipdb"
    _dataset(root, rows, set())
    for subject, _ in rows:
        eeg = root / subject / "eeg"
        eeg.mkdir(parents=True, exist_ok=True)
        stem = f"{subject}_task-block01_eeg"
        (eeg / f"{stem}.vhdr").write_text("Brain Vision Data Exchange Header File Version 1.0\n")
        (eeg / f"{stem}.eeg").write_bytes(b"binary companion")

    result = build_mipdb_inventory(
        root, protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )

    assert all(
        subject["recordings"] == [
            f"{subject['subject_id']}/eeg/{subject['subject_id']}_task-block01_eeg.vhdr"
        ]
        for subject in result["subjects"]
    )
    declared_paths = {
        item["path"] for item in result["acquisition_files"]
    }
    assert any(path.endswith("_eeg.vhdr") for path in declared_paths)
    assert any(path.endswith("_eeg.eeg") for path in declared_paths)
    assert all(
        len(item["sha256"]) == 64 and item["size_bytes"] >= 0
        for item in result["acquisition_files"]
    )


def test_inventory_identity_changes_when_raw_recording_bytes_change(
    tmp_path: Path,
) -> None:
    rows = [(f"sub-{index:03d}", "12") for index in range(12)]
    root = tmp_path / "mipdb"
    _dataset(root, rows, {subject for subject, _ in rows})

    first = build_mipdb_inventory(
        root, protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )
    recording = root / "sub-000/eeg/sub-000_task-rest_eeg.set"
    recording.write_bytes(b"changed-raw-recording")
    second = build_mipdb_inventory(
        root, protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )

    assert second["dataset_manifest_sha256"] != first["dataset_manifest_sha256"]


def test_acquisition_verifier_rejects_post_manifest_file_drift(tmp_path: Path) -> None:
    rows = [(f"sub-{index:03d}", "12") for index in range(12)]
    root = tmp_path / "mipdb"
    _dataset(root, rows, {subject for subject, _ in rows})
    manifest = build_mipdb_inventory(
        root, protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )

    mipdb_module.verify_mipdb_acquisition(root, manifest)
    (root / "sub-000/eeg/sub-000_task-rest_eeg.set").write_bytes(b"drift")

    with pytest.raises(MipdbInventoryError, match="acquisition.*differs"):
        mipdb_module.verify_mipdb_acquisition(root, manifest)


def test_inventory_separates_older_extrapolation_and_below_support(tmp_path: Path) -> None:
    rows = [(f"sub-{index:03d}", "12") for index in range(12)]
    rows += [("sub-older", "30"), ("sub-younger", "4")]
    _dataset(tmp_path / "mipdb", rows, {subject for subject, _ in rows})

    result = build_mipdb_inventory(
        tmp_path / "mipdb", protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
    )

    assert "sub-older" in result["cohorts"]["extrapolation"] or "sub-older" in result["cohorts"]["pilot"]
    if "sub-younger" not in result["cohorts"]["pilot"]:
        assert {row["subject_id"]: row["reason"] for row in result["exclusions"]}[
            "sub-younger"
        ] == "below_hbn_age_support"


def test_inventory_rejects_duplicate_subject_and_too_small_dataset(tmp_path: Path) -> None:
    duplicate_root = tmp_path / "duplicate"
    _dataset(
        duplicate_root,
        [("sub-001", "10"), ("sub-001", "11")],
        {"sub-001"},
    )
    with pytest.raises(MipdbInventoryError, match="duplicate"):
        build_mipdb_inventory(
            duplicate_root, protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
        )

    small_root = tmp_path / "small"
    rows = [(f"sub-{index:03d}", "10") for index in range(9)]
    _dataset(small_root, rows, {subject for subject, _ in rows})
    with pytest.raises(MipdbInventoryError, match="at least 10"):
        build_mipdb_inventory(
            small_root, protocol=PROTOCOL, hbn_age_support=(6.0, 18.0)
        )


def test_inventory_cli_writes_metadata_only_manifest(tmp_path: Path) -> None:
    rows = [(f"sub-{index:03d}", "12") for index in range(12)]
    dataset = tmp_path / "mipdb"
    _dataset(dataset, rows, {subject for subject, _ in rows})
    output = tmp_path / "inventory.json"
    hbn_manifest = tmp_path / "hbn.csv"
    hbn_manifest.write_text(
        "release,subject,age,recording_relpath,duration_s,split\n"
        "R1,sub-hbn-train-a,6.0,a.set,120.0,train\n"
        "R2,sub-hbn-train-b,18.0,b.set,120.0,train\n"
        "R8,sub-hbn-val,40.0,c.set,120.0,val\n"
        "R5,sub-hbn-test,50.0,d.set,120.0,test\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_mipdb_manifest.py"),
            "--protocol",
            str(ROOT / "configs" / "research" / "external_frozen_probe.json"),
            "--bids-root",
            str(dataset),
            "--hbn-manifest",
            str(hbn_manifest),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text())
    assert manifest["protocol_sha256"] == PROTOCOL.sha256
    assert manifest["hbn_age_support"]["minimum"] == 6.0
    assert manifest["hbn_age_support"]["maximum"] == 18.0
    assert len(manifest["hbn_age_support"]["source_manifest_sha256"]) == 64
    assert "metrics" not in manifest
    source = (ROOT / "scripts/build_mipdb_manifest.py").read_text(encoding="utf-8")
    assert 'add_argument("--hbn-age-min' not in source
    assert 'add_argument("--hbn-age-max' not in source


def test_finalize_cohort_cli_exposes_no_model_or_metric_controls(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    script_path = ROOT / "scripts/finalize_mipdb_cohort.py"
    spec = importlib.util.spec_from_file_location("finalize_mipdb_cohort", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "finalize_mipdb_cohort",
        lambda **kwargs: {
            "status": "finalized",
            "cohorts": {"pilot": [f"sub-{i}" for i in range(10)], "primary": ["sub-p"], "extrapolation": []},
            "underpowered": True,
            "cohort_qc_sha256": "a" * 64,
        },
    )

    assert module.main(
        [
            "--protocol", str(ROOT / "configs/research/external_frozen_probe.json"),
            "--bids-root", str((tmp_path / "bids").resolve()),
            "--draft-manifest", str((tmp_path / "draft.json").resolve()),
            "--qc-output", str((tmp_path / "qc.json").resolve()),
            "--output", str((tmp_path / "final.json").resolve()),
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "finalized"
    source = script_path.read_text(encoding="utf-8")
    assert 'add_argument("--device' not in source
    assert 'add_argument("--mapping' not in source
    assert 'add_argument("--metric' not in source
