from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pytest

from neurobench_age.data.screening_manifest import (
    DiscoveredRecording,
    MIN_RECORDING_SECONDS,
    ManifestRow,
    extend_manifest,
    read_manifest,
)


BASE_MANIFEST = Path(__file__).parents[1] / "results/canonical/data/age_medium_500_resting.csv"


def _synthetic_base() -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for release_number in range(1, 11):
        release = f"R{release_number}"
        split = "test" if release == "R5" else "val" if release == "R8" else "train"
        for index in range(50):
            rows.append(
                ManifestRow(
                    release=release,
                    subject=f"sub-base-{release_number:02d}-{index:03d}",
                    age=5.0 + index / 10.0,
                    recording_relpath=(
                        f"{release}/download/sub-base-{release_number:02d}-{index:03d}/eeg/"
                        f"sub-base-{release_number:02d}-{index:03d}_task-RestingState_eeg.set"
                    ),
                    duration_s=300.0,
                    split=split,
                )
            )
    return rows


def _synthetic_candidates() -> list[DiscoveredRecording]:
    rows: list[DiscoveredRecording] = []
    for release_number in range(1, 11):
        release = f"R{release_number}"
        for index in range(60):
            rows.append(
                DiscoveredRecording(
                    release=release,
                    subject=f"sub-added-{release_number:02d}-{index:03d}",
                    age=20.0 + index / 10.0,
                    recording_relpath=(
                        f"{release}/download/sub-added-{release_number:02d}-{index:03d}/eeg/"
                        f"sub-added-{release_number:02d}-{index:03d}_task-RestingState_eeg.set"
                    ),
                    duration_s=300.0,
                )
            )
    return rows


def test_repository_base_manifest_has_expected_shape() -> None:
    rows = read_manifest(BASE_MANIFEST)
    assert len(rows) == 500
    assert len({row.subject for row in rows}) == 500
    assert Counter(row.release for row in rows) == {f"R{i}": 50 for i in range(1, 11)}


def test_extension_is_nested_balanced_and_inherits_release_split() -> None:
    base = _synthetic_base()
    extended, audit = extend_manifest(base, _synthetic_candidates())

    assert len(extended) == 1000
    assert {row.recording_relpath for row in base} <= {
        row.recording_relpath for row in extended
    }
    assert len({row.subject for row in extended}) == 1000
    assert Counter(row.release for row in extended) == {f"R{i}": 100 for i in range(1, 11)}
    assert Counter(row.split for row in extended) == {
        "train": 800,
        "val": 100,
        "test": 100,
    }
    assert audit["nested_base_subjects"] is True
    assert audit["added_rows"] == 500


def test_extension_is_deterministic() -> None:
    base = _synthetic_base()
    candidates = _synthetic_candidates()
    first, first_audit = extend_manifest(base, candidates)
    second, second_audit = extend_manifest(base, list(reversed(candidates)))
    relabeled = [
        DiscoveredRecording(
            release=row.release,
            subject=row.subject,
            age=999.0 - row.age,
            recording_relpath=row.recording_relpath,
            duration_s=row.duration_s,
        )
        for row in candidates
    ]
    third, third_audit = extend_manifest(base, relabeled)

    assert first == second
    assert first_audit == second_audit
    assert [row.recording_relpath for row in first] == [
        row.recording_relpath for row in third
    ]
    assert third_audit["selection_rule"] == first_audit["selection_rule"]


def test_extension_rejects_non_nested_base_size() -> None:
    with pytest.raises(ValueError, match="exactly 500"):
        extend_manifest(_synthetic_base()[:-1], _synthetic_candidates())


def test_extension_fails_when_a_release_lacks_additions() -> None:
    candidates = [row for row in _synthetic_candidates() if row.release != "R3"]
    with pytest.raises(ValueError, match="release R3"):
        extend_manifest(_synthetic_base(), candidates)


def test_extension_rejects_duplicate_candidate_paths() -> None:
    candidates = _synthetic_candidates()
    candidates[1] = DiscoveredRecording(
        release=candidates[1].release,
        subject=candidates[0].subject,
        age=candidates[1].age,
        recording_relpath=candidates[0].recording_relpath,
        duration_s=candidates[1].duration_s,
    )
    with pytest.raises(ValueError, match="duplicate candidate recording path"):
        extend_manifest(_synthetic_base(), candidates)


@pytest.mark.parametrize("duration", [math.nan, math.inf, -math.inf])
def test_extension_excludes_nonfinite_candidate_duration(duration: float) -> None:
    candidates = [row for row in _synthetic_candidates() if row.release != "R1"]
    r1_candidates = _synthetic_candidates()[:50]
    r1_candidates[0] = DiscoveredRecording(
        release=r1_candidates[0].release,
        subject=r1_candidates[0].subject,
        age=r1_candidates[0].age,
        recording_relpath=r1_candidates[0].recording_relpath,
        duration_s=duration,
    )
    with pytest.raises(ValueError, match="release R1"):
        extend_manifest(_synthetic_base(), [*candidates, *r1_candidates])


def test_extension_rejects_release_path_mismatch() -> None:
    candidates = _synthetic_candidates()
    candidates[0] = DiscoveredRecording(
        release="R1",
        subject=candidates[0].subject,
        age=candidates[0].age,
        recording_relpath=candidates[0].recording_relpath.replace("R1/", "R2/", 1),
        duration_s=candidates[0].duration_s,
    )
    with pytest.raises(ValueError, match="does not belong to R1"):
        extend_manifest(_synthetic_base(), candidates)


def test_extension_rejects_malformed_base_recording_path() -> None:
    base = _synthetic_base()
    base[0] = ManifestRow(
        release=base[0].release,
        subject=base[0].subject,
        age=base[0].age,
        recording_relpath=base[0].recording_relpath.replace("R1/", "R2/", 1),
        duration_s=base[0].duration_s,
        split=base[0].split,
    )
    with pytest.raises(ValueError, match="does not belong to R1"):
        extend_manifest(base, _synthetic_candidates())


def test_extension_rejects_short_base_recording() -> None:
    base = _synthetic_base()
    base[0] = ManifestRow(
        release=base[0].release,
        subject=base[0].subject,
        age=base[0].age,
        recording_relpath=base[0].recording_relpath,
        duration_s=MIN_RECORDING_SECONDS,
        split=base[0].split,
    )
    with pytest.raises(ValueError, match="too short"):
        extend_manifest(base, _synthetic_candidates())
