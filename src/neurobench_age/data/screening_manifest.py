"""Build a deterministic 1000-subject manifest nested in the 500-subject run.

The script only prepares a screening data artifact. It does not train a model
or access the test loader. It is intended to run on the benchmark host where
the HBN recordings are available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


MANIFEST_FIELDS = (
    "release",
    "subject",
    "age",
    "recording_relpath",
    "duration_s",
    "split",
)
ALLOWED_SPLITS = frozenset({"train", "val", "test"})
MIN_RECORDING_SECONDS = 180.0
OFFICIAL_EXCLUDED_SUBJECTS = frozenset(
    {
        "sub-NDARWV769JM7",
        "sub-NDARME789TD2",
        "sub-NDARUA442ZVF",
        "sub-NDARJP304NK1",
        "sub-NDARTY128YLU",
        "sub-NDARDW550GU6",
        "sub-NDARLD243KRE",
        "sub-NDARUJ292JXV",
        "sub-NDARBA381JGH",
    }
)


@dataclass(frozen=True)
class ManifestRow:
    release: str
    subject: str
    age: float
    recording_relpath: str
    duration_s: float
    split: str


@dataclass(frozen=True)
class DiscoveredRecording:
    release: str
    subject: str
    age: float
    recording_relpath: str
    duration_s: float


def _release_number(release: str) -> int:
    if not release.startswith("R") or not release[1:].isdigit():
        raise ValueError(f"invalid release label: {release!r}")
    return int(release[1:])


def _sort_key(row: ManifestRow | DiscoveredRecording) -> tuple[str, str]:
    """Return a target-independent key that cannot use test labels."""

    return row.subject, row.recording_relpath


def _normalize_recording_path(*, release: str, subject: str, value: str) -> str:
    """Validate the official relative path shape and return its POSIX form."""

    raw = Path(value)
    if (
        raw.is_absolute()
        or not raw.parts
        or raw.parts[0] != release
        or ".." in raw.parts
    ):
        raise ValueError(
            f"recording path does not belong to {release}: {value!r}"
        )
    normalized = raw.as_posix()
    filename = raw.name
    if (
        raw.suffix != ".set"
        or "_task-RestingState_" not in filename
        or not filename.startswith(f"{subject}_")
        or len(raw.parts) != 5
        or raw.parts[1] != "download"
        or raw.parts[2] != subject
        or raw.parts[3] != "eeg"
    ):
        raise ValueError(f"recording path is not an official resting-state path: {value!r}")
    return normalized


def _midpoint_indices(candidate_count: int, selection_count: int) -> list[int]:
    """Return deterministic evenly spaced midpoint indices."""

    if selection_count <= 0 or candidate_count < selection_count:
        raise ValueError(
            f"cannot select {selection_count} rows from {candidate_count} candidates"
        )
    return [int((index + 0.5) * candidate_count / selection_count) for index in range(selection_count)]


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(
                f"manifest fields must be {MANIFEST_FIELDS}, got {reader.fieldnames}"
            )
        rows = []
        for row in reader:
            release = row["release"]
            subject = row["subject"]
            rows.append(
                ManifestRow(
                    release=release,
                    subject=subject,
                    age=float(row["age"]),
                    recording_relpath=_normalize_recording_path(
                        release=release,
                        subject=subject,
                        value=row["recording_relpath"],
                    ),
                    duration_s=float(row["duration_s"]),
                    split=row["split"],
                )
            )
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _canonical_bytes(rows: Sequence[ManifestRow]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=MANIFEST_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
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


def write_manifest(rows: Sequence[ManifestRow], path: Path) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest hash: {sidecar}")
    payload = _canonical_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar.write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def _validate_base_manifest(
    rows: Sequence[ManifestRow], *, target_subjects: int
) -> tuple[tuple[str, ...], dict[str, str]]:
    if len(rows) != 500:
        raise ValueError(f"nested extension requires exactly 500 base rows, got {len(rows)}")
    if target_subjects <= len(rows):
        raise ValueError("target subject count must be greater than the base manifest")
    releases = tuple(sorted({row.release for row in rows}, key=_release_number))
    if not releases or target_subjects % len(releases):
        raise ValueError("target subject count must divide evenly across base releases")
    base_counts = {release: sum(row.release == release for row in rows) for release in releases}
    if len(set(base_counts.values())) != 1:
        raise ValueError(f"base manifest is not release-balanced: {base_counts}")
    target_per_release = target_subjects // len(releases)
    if any(count > target_per_release for count in base_counts.values()):
        raise ValueError("target count is smaller than a base release count")

    subjects = [row.subject for row in rows]
    paths = [row.recording_relpath for row in rows]
    if len(set(subjects)) != len(subjects):
        raise ValueError("base manifest contains duplicate subjects")
    if len(set(paths)) != len(paths):
        raise ValueError("base manifest contains duplicate recording paths")
    for row in rows:
        if row.split not in ALLOWED_SPLITS:
            raise ValueError(f"invalid split in base manifest: {row.split!r}")
        if not math.isfinite(row.age) or not math.isfinite(row.duration_s):
            raise ValueError(f"non-finite base metadata for {row.subject}")
        if row.duration_s <= MIN_RECORDING_SECONDS:
            raise ValueError(f"base recording is too short for {row.subject}")
        if row.subject in OFFICIAL_EXCLUDED_SUBJECTS:
            raise ValueError(f"base manifest contains an officially excluded subject: {row.subject}")
        _normalize_recording_path(
            release=row.release,
            subject=row.subject,
            value=row.recording_relpath,
        )

    release_splits: dict[str, str] = {}
    for row in rows:
        previous = release_splits.setdefault(row.release, row.split)
        if previous != row.split:
            raise ValueError(f"base release has mixed split assignments: {row.release}")
    return releases, release_splits


def _eligible_candidates(
    candidates: Iterable[DiscoveredRecording],
    *,
    releases: Sequence[str],
    base_rows: Sequence[ManifestRow],
) -> list[DiscoveredRecording]:
    release_set = set(releases)
    base_subjects = {row.subject for row in base_rows}
    base_paths = {row.recording_relpath for row in base_rows}
    candidate_paths: set[str] = set()
    grouped: dict[tuple[str, str], list[DiscoveredRecording]] = {}
    for candidate in candidates:
        if candidate.release not in release_set:
            continue
        normalized_path = _normalize_recording_path(
            release=candidate.release,
            subject=candidate.subject,
            value=candidate.recording_relpath,
        )
        if normalized_path in candidate_paths:
            raise ValueError(f"duplicate candidate recording path: {normalized_path}")
        candidate_paths.add(normalized_path)
        if candidate.subject in OFFICIAL_EXCLUDED_SUBJECTS:
            continue
        if candidate.subject in base_subjects or normalized_path in base_paths:
            continue
        if not math.isfinite(candidate.age) or not math.isfinite(candidate.duration_s):
            continue
        if candidate.duration_s <= MIN_RECORDING_SECONDS:
            continue
        if normalized_path != candidate.recording_relpath:
            candidate = DiscoveredRecording(
                release=candidate.release,
                subject=candidate.subject,
                age=candidate.age,
                recording_relpath=normalized_path,
                duration_s=candidate.duration_s,
            )
        grouped.setdefault((candidate.release, candidate.subject), []).append(candidate)

    # A subject with multiple eligible recordings is ambiguous for this simple
    # nested manifest. Exclude it rather than silently choosing one run.
    unique_by_key = {
        key: rows[0]
        for key, rows in grouped.items()
        if len(rows) == 1
    }
    subject_counts: dict[str, int] = {}
    for _release, subject in unique_by_key:
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    return sorted(
        [
            candidate
            for (release, subject), candidate in unique_by_key.items()
            if subject_counts[subject] == 1
        ],
        key=lambda row: (_release_number(row.release), *_sort_key(row)),
    )


def extend_manifest(
    base_rows: Sequence[ManifestRow],
    candidates: Sequence[DiscoveredRecording],
    *,
    target_subjects: int = 1000,
) -> tuple[list[ManifestRow], dict[str, object]]:
    """Return a balanced nested manifest and a compact audit payload."""

    releases, release_splits = _validate_base_manifest(
        base_rows, target_subjects=target_subjects
    )
    eligible = _eligible_candidates(candidates, releases=releases, base_rows=base_rows)
    target_per_release = target_subjects // len(releases)
    base_counts = {release: sum(row.release == release for row in base_rows) for release in releases}
    selected: list[ManifestRow] = []
    eligible_by_release: dict[str, list[DiscoveredRecording]] = {release: [] for release in releases}
    for candidate in eligible:
        eligible_by_release[candidate.release].append(candidate)

    for release in releases:
        additional_count = target_per_release - base_counts[release]
        pool = sorted(eligible_by_release[release], key=_sort_key)
        if len(pool) < additional_count:
            raise ValueError(
                f"release {release} has only {len(pool)} eligible additions; "
                f"need {additional_count}"
            )
        for index in _midpoint_indices(len(pool), additional_count):
            candidate = pool[index]
            selected.append(
                ManifestRow(
                    release=candidate.release,
                    subject=candidate.subject,
                    age=candidate.age,
                    recording_relpath=candidate.recording_relpath,
                    duration_s=candidate.duration_s,
                    split=release_splits[release],
                )
            )

    combined = sorted(
        [*base_rows, *selected], key=lambda row: (_release_number(row.release), row.subject)
    )
    if len(combined) != target_subjects:
        raise AssertionError(f"internal target-size error: {len(combined)}")
    if len({row.subject for row in combined}) != target_subjects:
        raise AssertionError("internal duplicate-subject error")
    if len({row.recording_relpath for row in combined}) != target_subjects:
        raise AssertionError("internal duplicate-recording-path error")
    audit = {
        "selection_rule": "per-release sort by subject and recording path; midpoint indices",
        "base_rows": len(base_rows),
        "added_rows": len(selected),
        "eligible_unique_additions": len(eligible),
        "target_subjects": target_subjects,
        "release_counts": {
            release: sum(row.release == release for row in combined) for release in releases
        },
        "split_counts": {
            split: sum(row.split == split for row in combined)
            for split in ("train", "val", "test")
        },
        "release_to_split": release_splits,
        "nested_base_subjects": True,
    }
    return combined, audit


def discover_recordings(data_root: Path, releases: Sequence[str]) -> list[DiscoveredRecording]:
    """Discover eligible resting-state recordings using the official HBN layout."""

    try:
        import mne
    except Exception as exc:  # pragma: no cover - host dependency
        raise RuntimeError("MNE is required to discover HBN recordings") from exc

    discovered: list[DiscoveredRecording] = []
    for release in releases:
        download_root = data_root / release / "download"
        participants_path = download_root / "participants.tsv"
        ages: dict[str, float] = {}
        if participants_path.is_file():
            with participants_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    participant = row.get("participant_id")
                    raw_age = row.get("age", "")
                    try:
                        if participant and raw_age:
                            ages[participant] = float(raw_age)
                    except (TypeError, ValueError):
                        continue
        for path in sorted(download_root.glob("sub-*/eeg/*.set")):
            parts = path.stem.split("_")
            if len(parts) not in {3, 4} or parts[-1] != "eeg" or parts[1] != "task-RestingState":
                continue
            subject = parts[0]
            age = ages.get(subject)
            if age is None or not math.isfinite(age):
                continue
            raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
            duration_s = float(raw.n_times / raw.info["sfreq"])
            discovered.append(
                DiscoveredRecording(
                    release=release,
                    subject=subject,
                    age=age,
                    recording_relpath=path.resolve().relative_to(data_root.resolve()).as_posix(),
                    duration_s=duration_s,
                )
            )
    if not discovered:
        raise FileNotFoundError("no eligible HBN resting-state recordings discovered")
    return discovered


def build_manifest(
    *,
    base_manifest: Path,
    data_root: Path,
    output_manifest: Path,
    report_path: Path,
    target_subjects: int = 1000,
) -> dict[str, object]:
    base_rows = read_manifest(base_manifest)
    releases = tuple(sorted({row.release for row in base_rows}, key=_release_number))
    candidates = discover_recordings(data_root, releases)
    combined, audit = extend_manifest(
        base_rows, candidates, target_subjects=target_subjects
    )
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")
    output_digest = write_manifest(combined, output_manifest)
    base_digest = hashlib.sha256(base_manifest.read_bytes()).hexdigest()
    base_paths = {row.recording_relpath for row in base_rows}
    additions = [row for row in combined if row.recording_relpath not in base_paths]
    report: dict[str, object] = {
        "schema_version": 1,
        "data_mode": "historical_1000_subject_nested_screening",
        "base_manifest": str(base_manifest),
        "base_manifest_sha256": base_digest,
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": output_digest,
        "claim_scope": "screening_only_not_official_full_data_improvement",
        "base_release_count": len(releases),
        "added_subjects_sha256": hashlib.sha256(_canonical_bytes(additions)).hexdigest(),
        **audit,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-subjects", type=int, default=1000)
    args = parser.parse_args(argv)
    report = build_manifest(
        base_manifest=args.base_manifest,
        data_root=args.data_root,
        output_manifest=args.output_manifest,
        report_path=args.report,
        target_subjects=args.target_subjects,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
