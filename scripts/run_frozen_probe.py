#!/usr/bin/env python3
"""Train the exact frozen-REVE four-head by ten-seed HBN study matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.pipelines.frozen_probe_training import (
    FrozenEncoderError,
    load_frozen_probe_training_manifest,
    train_frozen_probe_study,
)
from neurobench_age.research.protocol import ProtocolError, load_study_protocol


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise FrozenEncoderError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise FrozenEncoderError("device must be auto, cpu, cuda, or mps")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    try:
        if not args.cache_root.is_absolute() or not args.output_root.is_absolute():
            raise FrozenEncoderError("cache-root and output-root must be absolute paths")
        repository_root = Path(__file__).resolve().parents[1]
        canonical_results = (repository_root / "results/canonical").resolve()
        if args.output_root.resolve().is_relative_to(canonical_results):
            raise FrozenEncoderError("output-root must be outside results/canonical")
        protocol = load_study_protocol(args.protocol)
        training_source_sha256 = source_tree_sha256(repository_root)
        records = load_frozen_probe_training_manifest(
            args.training_manifest, protocol=protocol
        )
        inventory = train_frozen_probe_study(
            records=records,
            cache_root=args.cache_root,
            output_root=args.output_root,
            training=protocol.training,
            device=_resolve_device(args.device),
            training_source_sha256=training_source_sha256,
            progress_sink=lambda event: print(
                json.dumps(event, sort_keys=True, allow_nan=False), flush=True
            ),
        )
    except (OSError, ProtocolError, FrozenEncoderError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "status": inventory["status"],
                "run_count": inventory["run_count"],
                "protocol_sha256": protocol.sha256,
                "training_source_sha256": training_source_sha256,
                "checkpoint_inventory_sha256": inventory[
                    "checkpoint_inventory_sha256"
                ],
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
