"""Materialize all R1--R10 resting-state files from OpenNeuro.

The official ``openneuro-py`` metadata query is retained, but file transfer is
performed with curl because some hosts cannot reliably download large S3 files
through the client's asynchronous HTTP transport. The operation is resumable:
completed files are verified by size and skipped on the next run.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


RELEASE_TO_DATASET = {
    f"R{number}": f"ds{dataset}"
    for number, dataset in (
        (1, "005505"),
        (2, "005506"),
        (3, "005507"),
        (4, "005508"),
        (5, "005509"),
        (6, "005510"),
        (7, "005511"),
        (8, "005512"),
        (9, "005514"),
        (10, "005515"),
    )
}
ROOT_METADATA = frozenset(
    {
        "participants.tsv",
        "participants.json",
        "task-RestingState_eeg.json",
        "dataset_description.json",
        "README",
        "CHANGES",
    }
)


@dataclass(frozen=True)
class RemoteFile:
    filename: str
    url: str
    size: int


def _selected(filename: str) -> bool:
    return filename in ROOT_METADATA or fnmatch.fnmatchcase(
        filename, "sub-*/eeg/*_task-RestingState*"
    )


def _metadata(dataset: str) -> tuple[str, list[RemoteFile]]:
    from openneuro._download import _get_download_metadata

    snapshot = _get_download_metadata(
        dataset_id=dataset,
        tag=None,
        max_retries=5,
        retry_backoff=1.0,
        metadata_timeout=60.0,
    )
    files: list[RemoteFile] = []
    for item in snapshot.files:
        if not _selected(item.filename):
            continue
        if not item.urls or item.size is None:
            raise ValueError(f"metadata has no URL/size for {item.filename}")
        files.append(
            RemoteFile(
                filename=item.filename,
                url=item.urls[0],
                size=int(item.size),
            )
        )
    if not any(file.filename.endswith("_task-RestingState_eeg.set") for file in files):
        raise ValueError(f"no resting-state recordings found in {dataset}")
    return snapshot.id, sorted(files, key=lambda file: file.filename)


def _download_one(remote: RemoteFile, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == remote.size:
        return "skipped"
    partial = destination.with_name(f".{destination.name}.part")
    partial.unlink(missing_ok=True)
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--output",
        str(partial),
        remote.url,
    ]
    try:
        subprocess.run(command, check=True)
        actual_size = partial.stat().st_size
        if actual_size != remote.size:
            raise RuntimeError(
                f"size mismatch for {remote.filename}: expected {remote.size}, got {actual_size}"
            )
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return "downloaded"


def _release_marker(root: Path, release: str) -> Path:
    return root / release / ".resting_direct_download.json"


def _marker_matches(marker: Path, snapshot_id: str, files: Sequence[RemoteFile], root: Path, release: str) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("snapshot_id") != snapshot_id or payload.get("file_count") != len(files):
        return False
    release_root = root / release / "download"
    return all(
        (release_root / remote.filename).is_file()
        and (release_root / remote.filename).stat().st_size == remote.size
        for remote in files
    )


def download_release(root: Path, release: str, workers: int) -> dict[str, Any]:
    dataset = RELEASE_TO_DATASET[release]
    snapshot_id, files = _metadata(dataset)
    marker = _release_marker(root, release)
    if _marker_matches(marker, snapshot_id, files, root, release):
        return {
            "release": release,
            "dataset": dataset,
            "snapshot_id": snapshot_id,
            "file_count": len(files),
            "total_bytes": sum(file.size for file in files),
            "status": "already_complete",
        }

    release_root = root / release / "download"
    counts = {"downloaded": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, remote, release_root / remote.filename): remote
            for remote in files
        }
        for future in as_completed(futures):
            result = future.result()
            counts[result] += 1
            if (counts["downloaded"] + counts["skipped"]) % 25 == 0:
                print(
                    f"{release}: {counts['downloaded'] + counts['skipped']}/{len(files)} files",
                    flush=True,
                )

    payload = {
        "schema_version": 1,
        "release": release,
        "dataset": dataset,
        "snapshot_id": snapshot_id,
        "file_count": len(files),
        "total_bytes": sum(file.size for file in files),
        "status": "completed",
        "files": [asdict(file) | {"path": file.filename} for file in files],
        "transfer": counts,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def download_all(root: Path, releases: Iterable[str], workers: int) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for release in releases:
        if release not in RELEASE_TO_DATASET:
            raise ValueError(f"unsupported release: {release}")
        summary = download_release(root, release, workers)
        summaries.append(summary)
        print(
            f"{release}: {summary['status']} {summary['file_count']} files "
            f"({summary['total_bytes']} bytes)",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "data_mode": "openneuro_resting_direct",
        "data_root": str(root.resolve()),
        "releases": summaries,
    }
    (root / "direct_download_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--releases", nargs="+", default=tuple(RELEASE_TO_DATASET))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(download_all(args.data_root, args.releases, args.workers), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
