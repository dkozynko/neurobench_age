"""Selective HBN resting-state acquisition helpers and CLI.

The module keeps acquisition separate from the official NeuralBench runtime.
Its pure helpers are intentionally independent of MNE and NeuralBench so the
download contract and provenance ordering can be tested without network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Callable

SELECTIVE_TASK = "task-RestingState"

RELEASE_TO_STUDY_ID: dict[str, str] = {
    "R1": "ds005505",
    "R2": "ds005506",
    "R3": "ds005507",
    "R4": "ds005508",
    "R5": "ds005509",
    "R6": "ds005510",
    "R7": "ds005511",
    "R8": "ds005512",
    "R9": "ds005514",
    "R10": "ds005515",
    "R11": "ds005516",
}

RELEASES = tuple(RELEASE_TO_STUDY_ID)
INCLUDE_PATTERNS = (
    "participants.tsv",
    "task-RestingState_eeg.json",
    "sub-*/eeg/*_task-RestingState*",
)
CURRENT_PROVENANCE_NAME = "selective_task_provenance.json"
CURRENT_DIGEST_NAME = "selective_task_provenance.sha256"
OFFICIAL_STUDY_NAME = "Shirazi2024Hbn"


def _ordered_releases(releases: Iterable[str]) -> tuple[str, ...]:
    """Validate release names and return them in official R1--R11 order."""

    values = tuple(releases)
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate releases are not allowed: {values}")
    unknown = sorted(set(values) - set(RELEASES))
    if unknown:
        raise ValueError(f"unknown HBN releases: {unknown}")
    requested = set(values)
    return tuple(release for release in RELEASES if release in requested)


def _normalize_relative_path(value: str | Path) -> str:
    """Normalize a relative inventory path to a POSIX spelling."""

    normalized = str(value).replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or normalized == "."
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _canonical_file_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate and sort inventory records by normalized relative path."""

    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        release = record.get("release")
        relative_path = _normalize_relative_path(record.get("relative_path", ""))
        size_bytes = record.get("size_bytes")
        if not isinstance(release, str) or release not in RELEASE_TO_STUDY_ID:
            raise ValueError(f"invalid file record release: {release!r}")
        if not relative_path.startswith(f"{release}/"):
            raise ValueError(f"file record has an invalid release path: {relative_path}")
        if not relative_path or relative_path in seen:
            raise ValueError(f"invalid or duplicate file path: {relative_path!r}")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError(f"invalid file size for {relative_path}: {size_bytes!r}")
        seen.add(relative_path)
        canonical.append(
            {
                "release": release,
                "relative_path": relative_path,
                "size_bytes": size_bytes,
            }
        )
    return tuple(sorted(canonical, key=lambda item: item["relative_path"]))


def _canonical_timeline_records(
    timelines: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Validate and sort timeline identities deterministically."""

    canonical: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    release_rank = {release: index for index, release in enumerate(RELEASES)}
    for timeline in timelines:
        release = timeline.get("release")
        subject = timeline.get("subject")
        task = timeline.get("task")
        run = timeline.get("run")
        if not isinstance(release, str) or release not in RELEASE_TO_STUDY_ID:
            raise ValueError(f"invalid timeline release: {release!r}")
        if not isinstance(subject, str) or not subject:
            raise ValueError(f"invalid timeline subject: {subject!r}")
        if task != SELECTIVE_TASK:
            raise ValueError(f"selective inventory contains non-resting task: {task!r}")
        if run is not None and not isinstance(run, str):
            raise ValueError(f"invalid timeline run: {run!r}")
        identity = (release, subject, task, run)
        if identity in seen:
            raise ValueError(f"duplicate timeline identity: {identity}")
        seen.add(identity)
        canonical.append(
            {
                "release": release,
                "subject": subject,
                "task": task,
                "run": run,
            }
        )
    return tuple(
        sorted(
            canonical,
            key=lambda item: (
                release_rank[item["release"]],
                item["subject"],
                item["task"],
                item["run"] is not None,
                item["run"] or "",
            ),
        )
    )


def _parse_timeline_name(path: Path) -> dict[str, Any]:
    """Parse an HBN EEGLAB stem into one official timeline identity."""

    parts = path.stem.split("_")
    if len(parts) == 3 and parts[-1] == "eeg":
        subject, task, _ = parts
        run = None
    elif len(parts) == 4 and parts[-1] == "eeg":
        subject, task, run, _ = parts
    else:
        raise ValueError(f"unsupported HBN EEG filename: {path.name}")
    if not subject.startswith("sub-") or not task.startswith("task-"):
        raise ValueError(f"unsupported HBN EEG filename: {path.name}")
    return {"subject": subject, "task": task, "run": run}


def _task_entity(path: Path) -> str | None:
    """Return a BIDS task entity from a subject EEG filename, if present."""

    for part in path.stem.split("_"):
        if part.startswith("task-"):
            return part
    return None


def _file_record(data_root: Path, path: Path, release: str) -> dict[str, Any]:
    """Build one normalized file inventory record."""

    relative = _normalize_relative_path(path.relative_to(data_root))
    return {
        "release": release,
        "relative_path": relative,
        "size_bytes": path.stat().st_size,
    }


def _set_contains_embedded_data(path: Path) -> bool:
    """Return whether an EEGLAB ``.set`` stores numeric data inline.

    HBN releases can contain both classic ``.set`` + external ``.fdt`` pairs
    and MATLAB ``.set`` files with a numeric ``data`` variable embedded in the
    file.  The latter are valid inputs for the official EEGLAB loader and must
    not be rejected as incomplete pairs.
    """

    try:
        from scipy.io import loadmat

        payload = loadmat(
            path,
            variable_names=("EEG", "data"),
            simplify_cells=True,
        )
    except Exception:
        return False

    candidates: list[Any] = [payload.get("data")]
    eeg = payload.get("EEG")
    if isinstance(eeg, Mapping):
        candidates.append(eeg.get("data"))
    for candidate in candidates:
        dtype = getattr(candidate, "dtype", None)
        size = getattr(candidate, "size", 0)
        if dtype is not None and getattr(dtype, "kind", None) in "biufc" and size > 0:
            return True
    return False


def _audit_release(data_root: Path, release: str) -> dict[str, Any]:
    """Audit one selective HBN release without mutating its files."""

    if release not in RELEASE_TO_STUDY_ID:
        raise ValueError(f"unknown HBN release: {release}")
    data_root = data_root.resolve()
    download_dir = data_root / release / "download"
    if not download_dir.is_dir():
        raise ValueError(f"release download directory is missing: {download_dir}")

    required_metadata = (
        download_dir / "participants.tsv",
        download_dir / f"{SELECTIVE_TASK}_eeg.json",
    )
    for path in required_metadata:
        if not path.is_file():
            raise ValueError(f"required metadata is missing: {path.name}")

    file_records = [_file_record(data_root, path, release) for path in required_metadata]
    set_files: dict[str, Path] = {}
    fdt_files: dict[str, Path] = {}
    timeline_records: list[dict[str, Any]] = []
    for eeg_dir in sorted(download_dir.glob("sub-*/eeg")):
        if not eeg_dir.is_dir():
            continue
        for path in sorted(item for item in eeg_dir.rglob("*") if item.is_file()):
            task = _task_entity(path)
            if task is None:
                continue
            if task != SELECTIVE_TASK:
                raise ValueError(f"non-resting task-bearing file: {path}")
            file_records.append(_file_record(data_root, path, release))
            if path.suffix == ".set":
                set_files[path.stem] = path
            elif path.suffix == ".fdt":
                fdt_files[path.stem] = path

    missing_fdt = sorted(
        stem
        for stem in set(set_files) - set(fdt_files)
        if not _set_contains_embedded_data(set_files[stem])
    )
    if missing_fdt:
        raise ValueError(f"missing .fdt companion for: {missing_fdt[0]}")
    orphan_fdt = sorted(set(fdt_files) - set(set_files))
    if orphan_fdt:
        raise ValueError(f"orphan .fdt file: {orphan_fdt[0]}")
    if not set_files:
        raise ValueError(f"no resting EEG recordings in release {release}")

    for path in sorted(set_files.values()):
        parsed = _parse_timeline_name(path)
        timeline_records.append({"release": release, **parsed})

    files = _canonical_file_records(file_records)
    timelines = _canonical_timeline_records(timeline_records)
    return {
        "release": release,
        "study_id": RELEASE_TO_STUDY_ID[release],
        "files": files,
        "timelines": timelines,
        "file_count": len(files),
        "timeline_count": len(timelines),
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one provenance payload using the fixed canonical encoding."""

    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _build_provenance_payload(
    *,
    data_root: Path,
    requested_releases: Iterable[str],
    audits: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    """Build deterministic aggregate provenance from audited releases."""

    requested = _ordered_releases(requested_releases)
    audit_by_release: dict[str, Mapping[str, Any]] = {}
    for audit in audits:
        release = audit.get("release")
        if not isinstance(release, str) or release not in RELEASE_TO_STUDY_ID:
            raise ValueError(f"invalid audited release: {release!r}")
        if release in audit_by_release:
            raise ValueError(f"duplicate audited release: {release}")
        audit_by_release[release] = audit

    audited_releases = tuple(release for release in RELEASES if release in audit_by_release)
    files = _canonical_file_records(
        record
        for release in audited_releases
        for record in audit_by_release[release]["files"]
    )
    timelines = _canonical_timeline_records(
        record
        for release in audited_releases
        for record in audit_by_release[release]["timelines"]
    )
    release_summaries = [
        {
            "release": release,
            "study_id": RELEASE_TO_STUDY_ID[release],
            "file_count": audit_by_release[release]["file_count"],
            "timeline_count": audit_by_release[release]["timeline_count"],
        }
        for release in audited_releases
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "data_mode": "selective_task",
        "task": SELECTIVE_TASK,
        "study": "Shirazi2024Hbn",
        "data_root": str(data_root.resolve()),
        "expected_releases": list(RELEASES),
        "requested_releases": list(requested),
        "audited_releases": release_summaries,
        "complete": audited_releases == RELEASES,
        "include_patterns": list(INCLUDE_PATTERNS),
        "files": [dict(record) for record in files],
        "timelines": [dict(record) for record in timelines],
        "file_count": len(files),
        "timeline_count": len(timelines),
    }
    return payload, _canonical_json_bytes(payload)


def _current_provenance_paths(data_root: Path) -> tuple[Path, Path]:
    """Return the mutable current acquisition JSON and digest sidecar paths."""

    root = Path(data_root).resolve()
    return root / CURRENT_PROVENANCE_NAME, root / CURRENT_DIGEST_NAME


def _include_digest() -> str:
    """Digest the fixed task/include contract used by custom markers."""

    contract = {"task": SELECTIVE_TASK, "include_patterns": list(INCLUDE_PATTERNS)}
    return hashlib.sha256(_canonical_json_bytes(contract)).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace one file atomically, cleaning up a temporary write on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _marker_path(data_root: Path, release: str) -> Path:
    """Return the marker path owned by this selective task/schema."""

    return (
        Path(data_root).resolve()
        / release
        / "download"
        / f".selective_task-RestingState-v1-{_include_digest()}.success.json"
    )


def _marker_payload(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compact release success marker payload."""

    return {
        "schema_version": 1,
        "release": audit["release"],
        "study_id": audit["study_id"],
        "task": SELECTIVE_TASK,
        "include_sha256": _include_digest(),
        "file_count": audit["file_count"],
        "timeline_count": audit["timeline_count"],
    }


def _marker_matches(path: Path, audit: Mapping[str, Any]) -> bool:
    """Return whether a custom marker exactly describes the current audit."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == _marker_payload(audit)


def _ensure_official_study_layout(data_root: Path) -> Path:
    """Expose the direct selective tree at NeuralFetch's study path.

    NeuralSet resolves ``DATA_DIR`` to ``DATA_DIR/Shirazi2024Hbn`` when the
    configured root is not already named after the study.  Selective
    acquisition intentionally stores releases directly under ``data_root``;
    this symlink preserves that contract without copying the recordings.
    """

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"data root is not a directory: {root}")
    study_path = root / OFFICIAL_STUDY_NAME
    if study_path.is_symlink():
        target = study_path.resolve(strict=False)
        if target != root:
            raise ValueError(
                f"official study link points to {target}, expected {root}"
            )
        return study_path
    if study_path.exists():
        if study_path.is_dir():
            return study_path
        raise ValueError(f"official study path is not a directory: {study_path}")
    study_path.symlink_to(root, target_is_directory=True)
    return study_path


def _default_openneuro_provider(
    study_id: str,
    target_dir: Path,
    include_patterns: tuple[str, ...],
    workers: int,
) -> None:
    """Download one release through the official OpenNeuro Python client."""

    import openneuro

    openneuro.download(
        dataset=study_id,
        target_dir=str(target_dir),
        include=list(include_patterns),
        max_concurrent_downloads=workers,
    )


def download_selective_hbn(
    *,
    data_root: Path,
    releases: Iterable[str] | None = None,
    workers: int = 5,
    provider: Callable[[str, Path, tuple[str, ...], int], None] | None = None,
) -> dict[str, Any]:
    """Download and audit the selected HBN resting-state releases."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"data root is not a directory: {root}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive integer: {workers!r}")
    requested = _ordered_releases(RELEASES if releases is None else releases)
    if not requested:
        raise ValueError("at least one release is required")
    download_provider = provider or _default_openneuro_provider
    provenance_path, digest_path = _current_provenance_paths(root)
    provenance_path.unlink(missing_ok=True)
    digest_path.unlink(missing_ok=True)

    try:
        for release in requested:
            marker = _marker_path(root, release)
            release_download = root / release / "download"
            release_download.mkdir(parents=True, exist_ok=True)
            audit: dict[str, Any] | None = None
            if marker.is_file():
                try:
                    audit = _audit_release(root, release)
                except ValueError:
                    audit = None
                if audit is not None and _marker_matches(marker, audit):
                    continue
                marker.unlink(missing_ok=True)

            download_provider(
                RELEASE_TO_STUDY_ID[release],
                release_download,
                INCLUDE_PATTERNS,
                workers,
            )
            audit = _audit_release(root, release)
            _atomic_write_bytes(
                marker,
                (_canonical_json_bytes(_marker_payload(audit))),
            )

        audits = []
        for release in RELEASES:
            release_download = root / release / "download"
            if release_download.exists():
                audits.append(_audit_release(root, release))
        payload, raw = _build_provenance_payload(
            data_root=root,
            requested_releases=requested,
            audits=audits,
        )
        _ensure_official_study_layout(root)
        _atomic_write_bytes(provenance_path, raw)
        _atomic_write_bytes(digest_path, (hashlib.sha256(raw).hexdigest() + "\n").encode("ascii"))
        return payload
    except BaseException:
        provenance_path.unlink(missing_ok=True)
        digest_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """Download and audit selective HBN resting-state releases."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--releases", nargs="+", default=list(RELEASES))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        payload = download_selective_hbn(
            data_root=args.data_root,
            releases=args.releases,
            workers=args.workers,
        )
    except Exception as error:
        parser.exit(1, f"selective HBN download failed: {error}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
