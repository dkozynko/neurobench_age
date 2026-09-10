#!/usr/bin/env python3
"""Cache frozen REVE representations for canonical HBN train/validation subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from neurobench_age.pipelines.frozen_probe import FrozenEncoderError
from neurobench_age.pipelines.independent import IndependentPipelineError
from neurobench_age.pipelines.representation_materialization import (
    materialize_hbn_representations,
)
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
    parser.add_argument(
        "--protocol-profile", choices=("primary", "layerwise"), default="primary"
    )
    parser.add_argument("--subject-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--preprocessing-cache-root", required=True, type=Path)
    parser.add_argument("--representation-cache-root", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=8)
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="+",
        default=None,
        help="optional subset of protocol layers to materialize",
    )
    parser.add_argument(
        "--pool-tokens",
        action="store_true",
        help="store arithmetic token means (exact for mean-linear probes)",
    )
    args = parser.parse_args(argv)

    try:
        paths = (
            args.subject_manifest,
            args.data_root,
            args.preprocessing_cache_root,
            args.representation_cache_root,
            args.training_manifest,
            args.mapping,
        )
        if any(not path.is_absolute() for path in paths):
            raise FrozenEncoderError("all data, cache, mapping, and output paths must be absolute")
        repository_root = Path(__file__).resolve().parents[1]
        canonical_results = (repository_root / "results/canonical").resolve()
        if any(
            path.resolve().is_relative_to(canonical_results)
            for path in (
                args.preprocessing_cache_root,
                args.representation_cache_root,
                args.training_manifest,
            )
        ):
            raise FrozenEncoderError(
                "cache and training-manifest outputs must be outside results/canonical"
            )
        protocol = load_study_protocol(args.protocol, profile=args.protocol_profile)
        manifest = materialize_hbn_representations(
            protocol=protocol,
            subject_manifest_path=args.subject_manifest,
            data_root=args.data_root,
            preprocessing_cache_root=args.preprocessing_cache_root,
            representation_cache_root=args.representation_cache_root,
            training_manifest_path=args.training_manifest,
            mapping_path=args.mapping,
            repository_root=repository_root,
            device=_resolve_device(args.device),
            extraction_batch_size=args.extraction_batch_size,
            materialized_layers=(
                None if args.layer_indices is None else tuple(args.layer_indices)
            ),
            pool_tokens=args.pool_tokens,
        )
    except (
        OSError,
        ProtocolError,
        IndependentPipelineError,
        FrozenEncoderError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "protocol_sha256": manifest["protocol_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "train_subjects": sum(
                    row["split"] == "train" for row in manifest["subjects"]
                ),
                "validation_subjects": sum(
                    row["split"] == "validation" for row in manifest["subjects"]
                ),
                "training_manifest": str(args.training_manifest.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
