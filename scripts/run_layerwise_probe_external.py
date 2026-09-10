#!/usr/bin/env python3
"""Run the lock-gated external evaluation for the layer-wise secondary study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.pipelines.frozen_probe import FrozenEncoderError
from neurobench_age.pipelines.layerwise_external import run_layerwise_external
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


def _resolve_training_source_sha256(checkpoint_inventory: Path) -> str:
    """Use the immutable source identity recorded at training time.

    External evaluation code may be repaired after training.  Its current
    source identity is recorded separately, while checkpoints remain bound to
    the source identity that created them.
    """

    try:
        payload = json.loads(Path(checkpoint_inventory).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenEncoderError(
            f"could not read checkpoint inventory: {checkpoint_inventory}"
        ) from error
    value = payload.get("training_source_sha256") if isinstance(payload, dict) else None
    if not isinstance(value, str) or len(value) != 64:
        raise FrozenEncoderError("checkpoint inventory has no valid training source identity")
    try:
        int(value, 16)
    except ValueError as error:
        raise FrozenEncoderError("checkpoint inventory has no valid training source identity") from error
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--training-protocol", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--primary-protocol", required=True, type=Path)
    parser.add_argument("--primary-training-manifest", required=True, type=Path)
    parser.add_argument("--primary-cache-root", required=True, type=Path)
    parser.add_argument("--mipdb-manifest", required=True, type=Path)
    parser.add_argument("--manifest-protocol", required=True, type=Path)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    try:
        paths = (
            args.protocol,
            args.training_protocol,
            args.checkpoint_root,
            args.checkpoint_inventory,
            args.training_manifest,
            args.mipdb_manifest,
            args.manifest_protocol,
            args.bids_root,
            args.mapping,
            args.cache_root,
            args.output_root,
        )
        if any(not path.is_absolute() for path in paths):
            raise FrozenEncoderError("all layer-wise data and artifact paths must be absolute")
        repository_root = Path(__file__).resolve().parents[1]
        protocol = load_study_protocol(args.protocol, profile="layerwise")
        manifest_protocol = load_study_protocol(args.manifest_protocol, profile="primary")
        primary_protocol = load_study_protocol(args.primary_protocol, profile="primary")
        training = load_frozen_probe_training_protocol(
            args.training_protocol, profile="layerwise"
        )
        if training.representation_protocol_sha256 != protocol.sha256:
            raise FrozenProbeTrainingProtocolError(
                "layer-wise training protocol does not match representation protocol"
            )
        result = run_layerwise_external(
            protocol=protocol,
            training=training,
            checkpoint_root=args.checkpoint_root,
            checkpoint_inventory_path=args.checkpoint_inventory,
            mipdb_manifest_path=args.mipdb_manifest,
            manifest_protocol_sha256=manifest_protocol.sha256,
            training_manifest_path=args.training_manifest,
            primary_protocol=primary_protocol,
            primary_training_manifest_path=args.primary_training_manifest,
            primary_cache_root=args.primary_cache_root,
            bids_root=args.bids_root,
            mapping_path=args.mapping,
            cache_root=args.cache_root,
            output_root=args.output_root,
            training_source_sha256=_resolve_training_source_sha256(
                args.checkpoint_inventory
            ),
            evaluation_source_sha256=source_tree_sha256(repository_root),
            device=_resolve_device(args.device),
        )
    except Exception as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "status": result["status"],
                "lock_sha256": result["lock"]["lock_sha256"],
                "metrics_sha256": result["metrics"]["metrics_sha256"],
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
