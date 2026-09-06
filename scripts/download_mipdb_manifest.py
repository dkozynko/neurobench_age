#!/usr/bin/env python3
"""Resumable download and verification for a MIPDB manifest subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ManifestFile:
    path: str
    url: str
    size: int
    checksum_algorithm: str
    checksum: str


def _read_files(path: Path) -> list[ManifestFile]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest must contain a non-empty files array")
    result: list[ManifestFile] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("manifest file entry must be an object")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {relative}")
        algorithm = str(item.get("checksum_algorithm", "sha256"))
        if algorithm not in {"sha256", "git"}:
            raise ValueError(f"unsupported checksum algorithm for {relative}")
        checksum = str(item["checksum"])
        expected_length = 64 if algorithm == "sha256" else 40
        if len(checksum) != expected_length or any(
            c not in "0123456789abcdef" for c in checksum.lower()
        ):
            raise ValueError(f"invalid {algorithm} checksum for {relative}")
        result.append(
            ManifestFile(
                path=relative.as_posix(),
                url=str(item["bytes_url"]),
                size=int(item["size"]),
                checksum_algorithm=algorithm,
                checksum=checksum.lower(),
            )
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches_checksum(path: Path, item: ManifestFile) -> bool:
    if item.checksum_algorithm == "sha256":
        return _sha256(path) == item.checksum
    return _git_blob_sha1(path) == item.checksum


def _fetch_one(item: ManifestFile, root: Path) -> str:
    destination = root / item.path
    partial = destination.with_name(f".{destination.name}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size == item.size:
        if _matches_checksum(destination, item):
            return "verified"
        destination.unlink()

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
        "--continue-at",
        "-",
        "--output",
        str(partial),
        item.url,
    ]
    subprocess.run(command, check=True)
    if partial.stat().st_size != item.size:
        raise RuntimeError(
            f"size mismatch for {item.path}: expected {item.size}, got {partial.stat().st_size}"
        )
    if not _matches_checksum(partial, item):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{item.checksum_algorithm} checksum mismatch for {item.path}")
    os.replace(partial, destination)
    return "downloaded"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")

    files = _read_files(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "verified": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_fetch_one, item, args.output_root): item for item in files}
        for index, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            counts[status] += 1
            if index == len(files) or index % 10 == 0:
                print(f"{index}/{len(files)} {counts}", flush=True)
    print(json.dumps({"files": len(files), "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
