#!/usr/bin/env python3
"""Train the predeclared layer-wise mean-linear secondary study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.pipelines.frozen_probe import FrozenEncoderError
from neurobench_age.pipelines.frozen_probe_training import (
    load_frozen_probe_training_manifest,
)
from neurobench_age.research.layerwise_probe import train_layerwise_probe_study
from neurobench_age.research.protocol import ProtocolError, load_study_protocol
from neurobench_age.research.training_protocol import (
    FrozenProbeTrainingProtocolError,
    load_frozen_probe_training_protocol,
)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise FrozenEncoderError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise FrozenEncoderError("device must be auto, cuda, or mps")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--training-protocol", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    try:
        paths = (
            args.protocol,
            args.training_protocol,
            args.training_manifest,
            args.cache_root,
            args.output_root,
        )
        if any(not path.is_absolute() for path in paths):
            raise FrozenEncoderError("protocol, cache, manifest, and output paths must be absolute")
        repository_root = Path(__file__).resolve().parents[1]
        protocol = load_study_protocol(args.protocol, profile="layerwise")
        training_protocol = load_frozen_probe_training_protocol(
            args.training_protocol, profile="layerwise"
        )
        if training_protocol.representation_protocol_sha256 != protocol.sha256:
            raise FrozenProbeTrainingProtocolError(
                "training protocol does not reference the layer-wise representation protocol"
            )
        records = load_frozen_probe_training_manifest(
            args.training_manifest, protocol=protocol
        )
        inventory = train_layerwise_probe_study(
            records=records,
            cache_root=args.cache_root,
            output_root=args.output_root,
            protocol=protocol,
            training=training_protocol,
            device=_resolve_device(args.device),
            training_source_sha256=source_tree_sha256(repository_root),
            progress_sink=lambda event: print(
                json.dumps(event, sort_keys=True, allow_nan=False), flush=True
            ),
        )
    except (
        OSError,
        ProtocolError,
        FrozenProbeTrainingProtocolError,
        FrozenEncoderError,
    ) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "status": inventory["status"],
                "run_count": inventory["run_count"],
                "protocol_sha256": protocol.sha256,
                "training_protocol_sha256": training_protocol.sha256,
                "training_source_sha256": source_tree_sha256(repository_root),
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
