#!/usr/bin/env python3
"""Seal the frozen-probe study after all manifests and checkpoints are ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.research.protocol import load_study_protocol
from neurobench_age.research.sealing import derive_study_payload
from neurobench_age.research.study_lock import (
    StudyLockError,
    seal_study,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--hbn-subject-manifest", required=True, type=Path)
    parser.add_argument("--hbn-training-manifest", required=True, type=Path)
    parser.add_argument("--hbn-data-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--mipdb-manifest", required=True, type=Path)
    parser.add_argument("--mipdb-bids-root", required=True, type=Path)
    parser.add_argument("--mipdb-pilot-qc", required=True, type=Path)
    parser.add_argument("--mipdb-cohort-qc", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        artifact_paths = (
            args.environment,
            args.hbn_subject_manifest,
            args.hbn_training_manifest,
            args.hbn_data_root,
            args.checkpoint_root,
            args.checkpoint_inventory,
            args.mipdb_manifest,
            args.mipdb_bids_root,
            args.mipdb_pilot_qc,
            args.mipdb_cohort_qc,
            args.output_root,
            args.lock,
        )
        if any(not path.is_absolute() for path in artifact_paths):
            raise StudyLockError("all artifact, data, output, and lock paths must be absolute")
        protocol = load_study_protocol(args.protocol)
        payload = derive_study_payload(
            protocol=protocol,
            repository_root=Path(__file__).resolve().parents[1],
            environment_path=args.environment,
            hbn_subject_manifest_path=args.hbn_subject_manifest,
            hbn_training_manifest_path=args.hbn_training_manifest,
            hbn_data_root=args.hbn_data_root,
            checkpoint_root=args.checkpoint_root,
            checkpoint_inventory_path=args.checkpoint_inventory,
            mipdb_manifest_path=args.mipdb_manifest,
            mipdb_bids_root=args.mipdb_bids_root,
            mipdb_pilot_qc_path=args.mipdb_pilot_qc,
            mipdb_cohort_qc_path=args.mipdb_cohort_qc,
            output_root=args.output_root,
        )
        lock = seal_study(args.lock, payload)
    except (OSError, ValueError, StudyLockError) as error:
        parser.error(str(error))
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
