from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from neurobench_age.data.selective_download import (
    INCLUDE_PATTERNS,
    RELEASE_TO_STUDY_ID,
    SELECTIVE_TASK,
    _audit_release,
    _build_provenance_payload,
    _canonical_file_records,
    _canonical_timeline_records,
    _current_provenance_paths,
    download_selective_hbn,
    _ensure_official_study_layout,
    _ordered_releases,
)


def test_selective_contract_has_exact_hbn_releases_and_task() -> None:
    assert SELECTIVE_TASK == "task-RestingState"
    assert list(RELEASE_TO_STUDY_ID.items()) == [
        ("R1", "ds005505"),
        ("R2", "ds005506"),
        ("R3", "ds005507"),
        ("R4", "ds005508"),
        ("R5", "ds005509"),
        ("R6", "ds005510"),
        ("R7", "ds005511"),
        ("R8", "ds005512"),
        ("R9", "ds005514"),
        ("R10", "ds005515"),
        ("R11", "ds005516"),
    ]
    assert "ds005513" not in RELEASE_TO_STUDY_ID.values()
    assert INCLUDE_PATTERNS == (
        "participants.tsv",
        "task-RestingState_eeg.json",
        "sub-*/eeg/*_task-RestingState*",
    )


def test_ordered_releases_uses_official_order_and_rejects_duplicates() -> None:
    assert _ordered_releases(["R10", "R1", "R2"]) == ("R1", "R2", "R10")

    try:
        _ordered_releases(["R1", "R1"])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate releases must be rejected")


def test_canonical_file_records_sort_by_normalized_relative_path() -> None:
    records = [
        {"release": "R1", "relative_path": "R1\\download\\b.set", "size_bytes": 2},
        {"release": "R1", "relative_path": "R1/download/participants.tsv", "size_bytes": 1},
        {"release": "R1", "relative_path": "R1/download/a.fdt", "size_bytes": 3},
    ]

    assert _canonical_file_records(records) == (
        {"release": "R1", "relative_path": "R1/download/a.fdt", "size_bytes": 3},
        {"release": "R1", "relative_path": "R1/download/b.set", "size_bytes": 2},
        {"release": "R1", "relative_path": "R1/download/participants.tsv", "size_bytes": 1},
    )


def test_canonical_file_records_reject_unsafe_or_cross_release_paths() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _canonical_file_records(
            [{"release": "R1", "relative_path": "../outside", "size_bytes": 1}]
        )
    with pytest.raises(ValueError, match="unsafe"):
        _canonical_file_records(
            [{"release": "R1", "relative_path": "/absolute/path", "size_bytes": 1}]
        )
    with pytest.raises(ValueError, match="release path"):
        _canonical_file_records(
            [{"release": "R1", "relative_path": "R2/download/file", "size_bytes": 1}]
        )


def test_canonical_timeline_records_sort_release_subject_task_and_null_run() -> None:
    timelines = [
        {"release": "R2", "subject": "sub-b", "task": SELECTIVE_TASK, "run": "run-2"},
        {"release": "R1", "subject": "sub-b", "task": SELECTIVE_TASK, "run": "run-1"},
        {"release": "R1", "subject": "sub-a", "task": SELECTIVE_TASK, "run": "run-1"},
        {"release": "R1", "subject": "sub-a", "task": SELECTIVE_TASK, "run": None},
    ]

    assert _canonical_timeline_records(timelines) == (
        {"release": "R1", "subject": "sub-a", "task": SELECTIVE_TASK, "run": None},
        {"release": "R1", "subject": "sub-a", "task": SELECTIVE_TASK, "run": "run-1"},
        {"release": "R1", "subject": "sub-b", "task": SELECTIVE_TASK, "run": "run-1"},
        {"release": "R2", "subject": "sub-b", "task": SELECTIVE_TASK, "run": "run-2"},
    )


def _make_release_tree(root: Path, release: str = "R1") -> Path:
    download = root / release / "download"
    eeg = download / "sub-NDAR001" / "eeg"
    eeg.mkdir(parents=True)
    (download / "participants.tsv").write_text("participant_id\tage\nsub-NDAR001\t10\n")
    (download / "task-RestingState_eeg.json").write_text('{"EEGReference": "Cz"}\n')
    (eeg / "sub-NDAR001_task-RestingState_eeg.set").write_bytes(b"set")
    (eeg / "sub-NDAR001_task-RestingState_eeg.fdt").write_bytes(b"fdt")
    return download


def test_audit_release_accepts_required_resting_tree_and_counts_all_files(tmp_path: Path) -> None:
    _make_release_tree(tmp_path)
    (tmp_path / "R1" / "download" / "README").write_text("allowed root metadata\n")

    audit = _audit_release(tmp_path, "R1")

    assert audit["release"] == "R1"
    assert audit["study_id"] == "ds005505"
    assert audit["file_count"] == 4
    assert audit["timeline_count"] == 1
    assert [record["relative_path"] for record in audit["files"]] == [
        "R1/download/participants.tsv",
        "R1/download/sub-NDAR001/eeg/sub-NDAR001_task-RestingState_eeg.fdt",
        "R1/download/sub-NDAR001/eeg/sub-NDAR001_task-RestingState_eeg.set",
        "R1/download/task-RestingState_eeg.json",
    ]


def test_official_study_layout_uses_zero_copy_compatibility_link(tmp_path: Path) -> None:
    _make_release_tree(tmp_path)

    study_path = _ensure_official_study_layout(tmp_path)

    assert study_path == tmp_path / "Shirazi2024Hbn"
    assert study_path.is_symlink()
    assert study_path.resolve() == tmp_path.resolve()
    assert (study_path / "R1" / "download" / "participants.tsv").is_file()


def test_audit_release_accepts_embedded_set_without_external_fdt(tmp_path: Path) -> None:
    download = _make_release_tree(tmp_path)
    eeg = download / "sub-NDAR001" / "eeg"
    (eeg / "sub-NDAR001_task-RestingState_eeg.fdt").unlink()
    savemat(
        eeg / "sub-NDAR001_task-RestingState_eeg.set",
        {
            "data": np.zeros((2, 3), dtype=np.float32),
            "setname": "sub-NDAR001_task-RestingState_eeg",
        },
    )

    audit = _audit_release(tmp_path, "R1")

    assert audit["file_count"] == 3
    assert audit["timeline_count"] == 1


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("missing_metadata", "participants.tsv"),
        ("empty", "no resting EEG recordings"),
        ("non_resting", "non-resting"),
        ("orphan_fdt", "orphan"),
        ("missing_fdt", "missing .fdt"),
    ],
)
def test_audit_release_rejects_invalid_trees_without_deleting_data(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    download = _make_release_tree(tmp_path)
    sentinel = download / "sentinel.txt"
    sentinel.write_text("preserve me\n")
    eeg = download / "sub-NDAR001" / "eeg"
    if mutation == "missing_metadata":
        (download / "participants.tsv").unlink()
    elif mutation == "empty":
        for path in eeg.iterdir():
            path.unlink()
    elif mutation == "non_resting":
        (eeg / "sub-NDAR001_task-contrastChangeDetection_eeg.set").write_bytes(b"set")
        (eeg / "sub-NDAR001_task-contrastChangeDetection_eeg.fdt").write_bytes(b"fdt")
    elif mutation == "orphan_fdt":
        (eeg / "sub-NDAR001_task-RestingState_eeg.set").unlink()
    elif mutation == "missing_fdt":
        (eeg / "sub-NDAR001_task-RestingState_eeg.fdt").unlink()

    with pytest.raises(ValueError, match=expected):
        _audit_release(tmp_path, "R1")
    assert sentinel.read_text() == "preserve me\n"


def test_provenance_is_deterministic_and_digest_covers_all_canonical_bytes(tmp_path: Path) -> None:
    _make_release_tree(tmp_path, "R1")
    _make_release_tree(tmp_path, "R2")
    audit_r1 = _audit_release(tmp_path, "R1")
    audit_r2 = _audit_release(tmp_path, "R2")

    first_payload, first_raw = _build_provenance_payload(
        data_root=tmp_path,
        requested_releases=("R2", "R1"),
        audits=(audit_r2, audit_r1),
    )
    second_payload, second_raw = _build_provenance_payload(
        data_root=tmp_path,
        requested_releases=("R1", "R2"),
        audits=(audit_r1, audit_r2),
    )

    assert first_payload == second_payload
    assert first_raw == second_raw
    assert first_payload["complete"] is False
    assert first_payload["file_count"] == 8
    assert first_payload["timeline_count"] == 2
    assert hashlib.sha256(first_raw).hexdigest() == hashlib.sha256(second_raw).hexdigest()


def test_selective_download_passes_exact_openneuro_arguments_and_writes_provenance(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    _make_release_tree(tmp_path, "R1")

    def fake_provider(
        study_id: str,
        target_dir: Path,
        include_patterns: tuple[str, ...],
        workers: int,
    ) -> None:
        calls.append(
            {
                "study_id": study_id,
                "target_dir": target_dir,
                "include_patterns": include_patterns,
                "workers": workers,
            }
        )

    payload = download_selective_hbn(
        data_root=tmp_path,
        releases=("R1",),
        workers=7,
        provider=fake_provider,
    )

    assert calls == [
        {
            "study_id": "ds005505",
            "target_dir": tmp_path / "R1" / "download",
            "include_patterns": INCLUDE_PATTERNS,
            "workers": 7,
        }
    ]
    assert payload["complete"] is False
    provenance_path, digest_path = _current_provenance_paths(tmp_path)
    assert provenance_path.is_file()
    assert digest_path.is_file()
    assert digest_path.read_text().strip() == hashlib.sha256(provenance_path.read_bytes()).hexdigest()


def test_selective_download_ignores_generic_full_marker_but_reuses_custom_marker(
    tmp_path: Path,
) -> None:
    download_dir = _make_release_tree(tmp_path, "R1")
    (download_dir / "openneuro_ds005505_success_download.txt").write_text("full\n")
    calls: list[str] = []

    def fake_provider(study_id: str, target_dir: Path, include_patterns: tuple[str, ...], workers: int) -> None:
        calls.append(study_id)

    download_selective_hbn(data_root=tmp_path, releases=("R1",), provider=fake_provider)
    download_selective_hbn(data_root=tmp_path, releases=("R1",), provider=fake_provider)

    assert calls == ["ds005505"]
    custom_markers = list(download_dir.glob(".selective_task-RestingState-v1-*.success.json"))
    assert len(custom_markers) == 1


def test_selective_download_rebuilds_partial_provenance_and_cleans_on_provider_failure(
    tmp_path: Path,
) -> None:
    _make_release_tree(tmp_path, "R1")
    _make_release_tree(tmp_path, "R2")
    provenance_path, digest_path = _current_provenance_paths(tmp_path)
    provenance_path.write_text('{"stale": true}\n')
    digest_path.write_text("stale\n")
    calls: list[str] = []

    def fake_provider(study_id: str, target_dir: Path, include_patterns: tuple[str, ...], workers: int) -> None:
        calls.append(study_id)
        if study_id == "ds005506":
            raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        download_selective_hbn(
            data_root=tmp_path,
            releases=("R1", "R2"),
            provider=fake_provider,
        )

    assert calls == ["ds005505", "ds005506"]
    assert not provenance_path.exists()
    assert not digest_path.exists()
    assert (tmp_path / "R1" / "download" / "participants.tsv").exists()
