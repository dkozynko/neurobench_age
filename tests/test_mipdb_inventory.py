from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

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
            "--hbn-age-min",
            "6",
            "--hbn-age-max",
            "18",
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
    assert "metrics" not in manifest
