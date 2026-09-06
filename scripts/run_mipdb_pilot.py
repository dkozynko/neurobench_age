#!/usr/bin/env python3
"""Run target-free preprocessing and frozen-encoder QC on the MIPDB pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from neurobench_age.data.mipdb import MipdbPreprocessingError
from neurobench_age.pipelines.frozen_probe import FrozenEncoderError
from neurobench_age.pipelines.representation_materialization import run_mipdb_pilot
from neurobench_age.research.protocol import ProtocolError, load_study_protocol


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise FrozenEncoderError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise FrozenEncoderError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise FrozenEncoderError("device must be auto, cpu, cuda, or mps")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--mipdb-manifest", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        paths = (
            args.bids_root,
            args.mipdb_manifest,
            args.mapping,
            args.output,
        )
        if any(not path.is_absolute() for path in paths):
            raise FrozenEncoderError(
                "BIDS, manifest, mapping, and output paths must be absolute"
            )
        protocol = load_study_protocol(args.protocol)
        report = run_mipdb_pilot(
            protocol=protocol,
            bids_root=args.bids_root,
            manifest_path=args.mipdb_manifest,
            mapping_path=args.mapping,
            output_path=args.output,
            device=_resolve_device(args.device),
            extraction_batch_size=args.extraction_batch_size,
        )
    except (
        OSError,
        ProtocolError,
        MipdbPreprocessingError,
        FrozenEncoderError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "status": report["status"],
                "pilot_subject_count": report["pilot_subject_count"],
                "protocol_sha256": report["protocol_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
