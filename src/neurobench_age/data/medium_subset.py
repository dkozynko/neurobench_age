"""Deterministic 100-subject manifest and parity helpers for Age experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


RELEASE_ORDER = tuple(f"R{i}" for i in range(1, 11))
MANIFEST_FIELDS = (
    "release",
    "subject",
    "age",
    "recording_relpath",
    "duration_s",
    "split",
)
WINDOW_KEY_FIELDS = (
    "recording_relpath",
    "release",
    "subject",
    "split",
    "start_sample",
    "n_samples",
)


@dataclass(frozen=True)
class SelectedRecording:
    """One canonical resting-state recording in the Medium subset."""

    release: str
    subject: str
    age: float
    recording_relpath: str
    duration_s: float
    split: str


@dataclass(frozen=True)
class WindowKey:
    """Stable identity of one 2-second Age window."""

    recording_relpath: str
    release: str
    subject: str
    split: str
    start_sample: int
    n_samples: int

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.recording_relpath,
            self.release,
            self.subject,
            self.split,
            self.start_sample,
            self.n_samples,
        )


def numeric_release_order(releases: Iterable[str]) -> list[str]:
    """Return unique release labels in numeric order."""

    unique = list(dict.fromkeys(releases))
    invalid = [release for release in unique if release not in RELEASE_ORDER]
    if invalid:
        raise ValueError(f"unsupported releases: {sorted(invalid)}")
    return sorted(unique, key=lambda release: int(release[1:]))


def select_midpoint_indices(candidate_count: int, selection_count: int) -> list[int]:
    """Select evenly spaced zero-based midpoint indices deterministically."""

    if candidate_count < selection_count or selection_count <= 0:
        raise ValueError(f"cannot select {selection_count} rows from {candidate_count} candidates")
    return [
        math.floor((index + 0.5) * candidate_count / selection_count)
        for index in range(selection_count)
    ]


def split_releases(
    releases: Sequence[str],
    *,
    test_release: str = "R5",
    validation_ratio: float = 0.091,
    random_state: int = 33,
) -> dict[str, str]:
    """Apply the Age task's release-level train/validation/test split."""

    ordered = numeric_release_order(releases)
    if test_release not in ordered:
        raise ValueError(f"test release {test_release!r} is missing")
    non_test = [release for release in ordered if release != test_release]
    if not non_test:
        raise ValueError("at least one non-test release is required")

    permutation = np.random.RandomState(random_state).permutation(len(non_test))
    n_validation = max(1, int(math.ceil(len(non_test) * validation_ratio)))
    validation = {non_test[index] for index in permutation[:n_validation]}
    return {
        release: (
            "test"
            if release == test_release
            else "val"
            if release in validation
            else "train"
        )
        for release in ordered
    }


def _recording_relpath(path: Path, data_root: Path) -> str:
    """Return a normalized POSIX path relative to the HBN data root."""

    try:
        relative = path.resolve().relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(f"recording is outside data root: {path}") from exc
    return relative.as_posix()


def _candidate_key(recording: Any, data_root: Path) -> tuple[float, str, str]:
    age = float(recording.age)
    return age, str(recording.subject), _recording_relpath(recording.path, data_root)


def _is_eligible(recording: Any) -> bool:
    return (
        recording.release in RELEASE_ORDER
        and recording.task == "task-RestingState"
        and recording.path.suffix == ".set"
        and recording.age is not None
        and np.isfinite(float(recording.age))
        and float(recording.duration_s) > 180.0
    )


def _eligible_candidates(recordings: Iterable[Any]) -> list[Any]:
    grouped: dict[tuple[str, str], list[Any]] = {}
    for recording in recordings:
        if _is_eligible(recording):
            grouped.setdefault((recording.release, recording.subject), []).append(recording)
    return [rows[0] for rows in grouped.values() if len(rows) == 1]


def validate_selected_recordings(
    recordings: Sequence[Any], *, data_root: Path
) -> None:
    """Fail closed on the selected-recording invariants."""

    keys = [(row.release, row.subject) for row in recordings]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (release, subject) key")
    subjects = [row.subject for row in recordings]
    if len(set(subjects)) != len(subjects):
        raise ValueError("duplicate subject across releases")
    if len(recordings) != 100:
        raise ValueError(f"expected 100 selected recordings, got {len(recordings)}")
    for row in recordings:
        if not _is_eligible(row):
            raise ValueError(f"selected recording is not Age-eligible: {row}")
        if _recording_relpath(row.path, data_root) != row.recording_relpath:
            raise ValueError(f"recording path mismatch for {row.subject}")
    test_ages = {float(row.age) for row in recordings if row.split == "test"}
    if len(test_ages) < 20:
        raise ValueError(f"test set needs at least 20 distinct ages, got {len(test_ages)}")


def build_medium_manifest(
    recordings: Sequence[Any], *, data_root: Path
) -> list[SelectedRecording]:
    """Build the deterministic 100-subject Medium manifest."""

    candidates = _eligible_candidates(recordings)
    by_release: dict[str, list[Any]] = {release: [] for release in RELEASE_ORDER}
    for recording in candidates:
        by_release[recording.release].append(recording)
    for release in by_release:
        by_release[release].sort(key=lambda row: _candidate_key(row, data_root))

    selected: list[Any] = []
    r5_bands = np.array_split(by_release["R5"], 4)
    for band in r5_bands:
        rows = list(band)
        indices = select_midpoint_indices(len(rows), 10)
        selected.extend(rows[index] for index in indices)

    train_releases = [release for release in RELEASE_ORDER if release != "R5"]
    quotas = {release: 6 + int(index < 6) for index, release in enumerate(train_releases)}
    for release in train_releases:
        rows = by_release[release]
        selected.extend(rows[index] for index in select_midpoint_indices(len(rows), quotas[release]))

    assignments = split_releases(RELEASE_ORDER)
    result = [
        SelectedRecording(
            release=row.release,
            subject=row.subject,
            age=float(row.age),
            recording_relpath=_recording_relpath(row.path, data_root),
            duration_s=float(row.duration_s),
            split=assignments[row.release],
        )
        for row in selected
    ]
    result.sort(key=lambda row: (int(row.release[1:]), row.subject))
    validate_selected_recordings(
        [
            next(recording for recording in recordings if _recording_relpath(recording.path, data_root) == row.recording_relpath)
            for row in result
        ],
        data_root=data_root,
    )
    return result


def _canonical_manifest_bytes(rows: Sequence[SelectedRecording]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "release": row.release,
                "subject": row.subject,
                "age": f"{row.age:.6f}",
                "recording_relpath": row.recording_relpath,
                "duration_s": f"{row.duration_s:.6f}",
                "split": row.split,
            }
        )
    return buffer.getvalue().encode("utf-8")


def write_manifest(rows: Sequence[SelectedRecording], path: Path) -> str:
    """Write canonical manifest bytes and return their SHA-256."""

    payload = _canonical_manifest_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> list[SelectedRecording]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and tuple(rows[0]) != MANIFEST_FIELDS:
        raise ValueError(f"manifest fields must be {MANIFEST_FIELDS}")
    return [
        SelectedRecording(
            release=row["release"],
            subject=row["subject"],
            age=float(row["age"]),
            recording_relpath=Path(row["recording_relpath"]).as_posix(),
            duration_s=float(row["duration_s"]),
            split=row["split"],
        )
        for row in rows
    ]


def filter_recordings_by_manifest(
    recordings: Sequence[Any],
    manifest: Sequence[SelectedRecording],
    *,
    data_root: Path,
) -> list[Any]:
    """Filter discovered recordings and validate every manifest row."""

    by_path = {row.recording_relpath: row for row in manifest}
    if len(by_path) != len(manifest):
        raise ValueError("manifest contains duplicate recording paths")
    found: dict[str, Any] = {}
    for recording in recordings:
        relative = _recording_relpath(recording.path, data_root)
        expected = by_path.get(relative)
        if expected is None:
            continue
        if (
            recording.release != expected.release
            or recording.subject != expected.subject
            or recording.age is None
            or not np.isclose(float(recording.age), expected.age, atol=1e-6, rtol=0.0)
            or not np.isclose(float(recording.duration_s), expected.duration_s, atol=1e-6, rtol=0.0)
            or not _is_eligible(recording)
        ):
            raise ValueError(f"metadata mismatch for {relative}")
        found[relative] = recording

    missing = sorted(set(by_path).difference(found))
    if missing:
        raise ValueError(f"manifest recordings were not discovered: {missing}")
    return [found[row.recording_relpath] for row in manifest]


def build_window_keys(
    recordings: Sequence[Any], *, data_root: Path
) -> list[WindowKey]:
    """Build integer sample identities for all 60 Age windows per recording."""

    keys: list[WindowKey] = []
    for row in recordings:
        relative = getattr(row, "recording_relpath", None)
        if relative is None:
            relative = _recording_relpath(row.path, data_root)
        split = str(row.split)
        for index in range(60):
            keys.append(
                WindowKey(
                    recording_relpath=str(relative),
                    release=str(row.release),
                    subject=str(row.subject),
                    split=split,
                    start_sample=12000 + 400 * index,
                    n_samples=400,
                )
            )
    return keys


def _as_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, WindowKey):
        return value.as_tuple()
    if isinstance(value, tuple):
        return value
    return tuple(value)


def compare_manifest_keys(left: Sequence[Any], right: Sequence[Any]) -> bool:
    """Compare key collections order-independently and reject duplicates."""

    left_keys = [_as_key(value) for value in left]
    right_keys = [_as_key(value) for value in right]
    if len(set(left_keys)) != len(left_keys) or len(set(right_keys)) != len(right_keys):
        return False
    return set(left_keys) == set(right_keys)


def _select_command(args: argparse.Namespace) -> int:
    from ..pipelines.independent import discover_hbn_recordings

    data_root = args.data_root.resolve()
    recordings = discover_hbn_recordings(data_root)
    manifest = build_medium_manifest(recordings, data_root=data_root)
    digest = write_manifest(manifest, args.output)
    report = {
        "manifest": str(args.output),
        "manifest_sha256": digest,
        "rows": len(manifest),
        "split_counts": {
            split: sum(row.split == split for row in manifest)
            for split in ("train", "val", "test")
        },
        "test_subjects": sum(row.split == "test" for row in manifest),
        "test_distinct_ages": len({row.age for row in manifest if row.split == "test"}),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--data-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "select":
        return _select_command(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
