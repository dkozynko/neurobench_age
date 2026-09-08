#!/usr/bin/env python3
"""Run the sealed MIPDB primary holdout once from frozen representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence

import torch

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.pipelines.external_holdout import (
    ExternalHoldoutError,
    RuntimeProvenance,
    run_external_holdout,
)
from neurobench_age.pipelines.representation_materialization import (
    LazyMipdbRepresentationProvider,
)
from neurobench_age.research.protocol import load_study_protocol
from neurobench_age.research.study_lock import (
    StudyLockError,
    load_study_lock,
    record_resumable_failure,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_provenance(
    repository_root: Path, environment_path: Path
) -> RuntimeProvenance:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ExternalHoldoutError("could not establish exact Git provenance") from error
    if not revision:
        raise ExternalHoldoutError("Git revision is empty")
    return RuntimeProvenance(
        training_source_sha256=source_tree_sha256(repository_root),
        git_revision=revision,
        git_dirty=dirty,
        environment_sha256=_sha256_file(environment_path),
    )


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ExternalHoldoutError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ExternalHoldoutError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ExternalHoldoutError("device must be auto, cpu, cuda, or mps")
    return requested


def _validate_protocol_and_power(
    *, protocol: object, lock: dict[str, object], manifest_path: Path
) -> dict[str, object]:
    if lock.get("protocol_sha256") != getattr(protocol, "sha256", None):
        raise ExternalHoldoutError("sealed lock does not match --protocol")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        primary = manifest["cohorts"]["primary"]
        minimum = protocol.datasets.minimum_primary_subjects
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        raise ExternalHoldoutError("could not validate primary cohort power") from error
    if not isinstance(primary, list):
        raise ExternalHoldoutError("MIPDB primary cohort must be an ordered array")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum <= 0
    ):
        raise ExternalHoldoutError("protocol minimum primary cohort size is invalid")
    underpowered = len(primary) < minimum
    if (
        manifest.get("minimum_primary_subjects") != minimum
        or manifest.get("underpowered") is not underpowered
    ):
        raise ExternalHoldoutError(
            "MIPDB power metadata does not match the protocol and primary cohort"
        )
    return {
        "primary_subjects": len(primary),
        "minimum_primary_subjects": minimum,
        "underpowered": underpowered,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--mipdb-manifest", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        path_arguments = (
            args.lock,
            args.checkpoint_root,
            args.checkpoint_inventory,
            args.mipdb_manifest,
            args.environment,
            args.cache_root,
            args.bids_root,
            args.mapping,
            args.output_root,
        )
        if any(not path.is_absolute() for path in path_arguments):
            raise ExternalHoldoutError(
                "lock, artifact, cache, and output paths must be absolute"
            )
        repository_root = Path(__file__).resolve().parents[1]
        protocol = load_study_protocol(args.protocol)
        lock = load_study_lock(args.lock)
        power = _validate_protocol_and_power(
            protocol=protocol, lock=lock, manifest_path=args.mipdb_manifest
        )
        runtime = _runtime_provenance(repository_root, args.environment)
        device = _resolve_device(args.device)
        representation_provider = LazyMipdbRepresentationProvider(
            protocol=protocol,
            bids_root=args.bids_root,
            manifest_path=args.mipdb_manifest,
            cache_root=args.cache_root,
            mapping_path=args.mapping,
            started_marker_path=args.output_root / "evaluation_started.json",
            expected_lock_sha256=lock["lock_sha256"],
            device=device,
            extraction_batch_size=args.extraction_batch_size,
        )
        inventory = run_external_holdout(
            lock_path=args.lock,
            checkpoint_root=args.checkpoint_root,
            inventory_path=args.checkpoint_inventory,
            mipdb_manifest_path=args.mipdb_manifest,
            environment_path=args.environment,
            output_root=args.output_root,
            runtime=runtime,
            representation_provider=representation_provider,
            device=device,
        )
    except Exception as error:
        try:
            record_resumable_failure(args.lock, error=str(error))
        except (OSError, StudyLockError):
            pass
        parser.error(str(error))

    print(
        json.dumps(
            {
                "status": inventory["status"],
                "prediction_count": inventory["prediction_count"],
                "prediction_inventory_sha256": inventory[
                    "prediction_inventory_sha256"
                ],
                "primary_subjects": power["primary_subjects"],
                "minimum_primary_subjects": power["minimum_primary_subjects"],
                "underpowered": power["underpowered"],
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
