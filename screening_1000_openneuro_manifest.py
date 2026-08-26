"""Build the nested 1000-subject manifest from OpenNeuro metadata only.

The downloader must not materialize every HBN recording just to choose a
screening subset.  This module reads participant ages and per-recording
``RecordingDuration`` from the small OpenNeuro sidecars, then delegates the
deterministic nested selection to ``screening_1000_manifest``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from screening_1000_manifest import (
    DiscoveredRecording,
    _canonical_bytes,
    extend_manifest,
    read_manifest,
    write_manifest,
)


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


def _download_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


def _participant_ages(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    ages: dict[str, float] = {}
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        subject = row.get("participant_id")
        raw_age = row.get("age")
        try:
            age = float(raw_age) if raw_age is not None else math.nan
        except (TypeError, ValueError):
            continue
        if subject and math.isfinite(age):
            ages[subject] = age
    return ages


def candidate_from_metadata(
    *, release: str, filename: str, age: float, sidecar: dict[str, Any]
) -> DiscoveredRecording:
    parts = filename.split("/")
    if (
        len(parts) != 3
        or parts[0].startswith("sub-") is False
        or parts[1] != "eeg"
        or not parts[2].endswith("_task-RestingState_eeg.json")
    ):
        raise ValueError(f"not a resting-state sidecar: {filename!r}")
    subject = parts[0]
    duration = sidecar.get("RecordingDuration")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        raise ValueError(f"sidecar has no numeric RecordingDuration: {filename!r}")
    return DiscoveredRecording(
        release=release,
        subject=subject,
        age=float(age),
        recording_relpath=f"{release}/download/{filename.removesuffix('.json')}.set",
        duration_s=float(duration),
    )


def discover_openneuro_candidates(
    releases: Iterable[str],
) -> tuple[list[DiscoveredRecording], dict[str, str]]:
    from openneuro._download import _get_download_metadata

    candidates: list[DiscoveredRecording] = []
    snapshots: dict[str, str] = {}
    for release in releases:
        dataset = RELEASE_TO_DATASET[release]
        snapshot = _get_download_metadata(
            dataset_id=dataset,
            tag=None,
            max_retries=5,
            retry_backoff=1.0,
            metadata_timeout=60.0,
        )
        snapshots[release] = snapshot.id
        by_name = {item.filename: item for item in snapshot.files}
        participants = by_name.get("participants.tsv")
        if participants is None or not participants.urls:
            raise ValueError(f"{release} metadata has no participants.tsv")
        ages = _participant_ages(participants.urls[0])
        sidecars: list[tuple[str, float, str]] = []
        for item in snapshot.files:
            filename = item.filename
            if not filename.startswith("sub-") or "/eeg/" not in filename:
                continue
            if not filename.endswith("_task-RestingState_eeg.json"):
                continue
            subject = filename.split("/", 1)[0]
            age = ages.get(subject)
            if age is None or not item.urls:
                continue
            sidecars.append((filename, age, item.urls[0]))

        def load_candidate(
            item: tuple[str, float, str],
        ) -> DiscoveredRecording | None:
            filename, age, url = item
            try:
                sidecar = _download_json(url)
                return candidate_from_metadata(
                    release=release,
                    filename=filename,
                    age=age,
                    sidecar=sidecar,
                )
            except (TypeError, ValueError, json.JSONDecodeError, OSError):
                return None

        # Sidecars are tiny, independent requests.  Bounded concurrency keeps
        # manifest construction fast while the per-request timeout prevents a
        # single unavailable S3 endpoint from stalling the whole screen.
        with ThreadPoolExecutor(max_workers=32) as executor:
            for candidate in executor.map(load_candidate, sidecars):
                if candidate is not None:
                    candidates.append(candidate)
    return candidates, snapshots


def build_manifest_from_openneuro(
    *,
    base_manifest: Path,
    output_manifest: Path,
    report_path: Path,
    target_subjects: int = 1000,
) -> dict[str, Any]:
    base_rows = read_manifest(base_manifest)
    releases = tuple(sorted({row.release for row in base_rows}, key=lambda value: int(value[1:])))
    candidates, snapshots = discover_openneuro_candidates(releases)
    combined, audit = extend_manifest(
        base_rows,
        candidates,
        target_subjects=target_subjects,
    )
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")
    output_digest = write_manifest(combined, output_manifest)
    base_digest = hashlib.sha256(base_manifest.read_bytes()).hexdigest()
    base_paths = {row.recording_relpath for row in base_rows}
    additions = [row for row in combined if row.recording_relpath not in base_paths]
    report: dict[str, Any] = {
        "schema_version": 1,
        "data_mode": "historical_1000_subject_nested_screening",
        "selection_source": "openneuro_metadata_sidecars",
        "base_manifest": str(base_manifest),
        "base_manifest_sha256": base_digest,
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": output_digest,
        "candidate_count": len(candidates),
        "added_subjects_sha256": hashlib.sha256(_canonical_bytes(additions)).hexdigest(),
        "openneuro_snapshots": snapshots,
        "claim_scope": "screening_only_not_official_full_data_improvement",
        **audit,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-subjects", type=int, default=1000)
    args = parser.parse_args(argv)
    report = build_manifest_from_openneuro(
        base_manifest=args.base_manifest,
        output_manifest=args.output_manifest,
        report_path=args.report,
        target_subjects=args.target_subjects,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
