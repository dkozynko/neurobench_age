#!/usr/bin/env python3
"""Combine existing final-layer caches with compact pooled early-layer caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from neurobench_age.pipelines.frozen_probe import (
    FrozenEncoderError,
    inspect_cached_representation_metadata,
    load_cached_representations,
    write_cached_representations,
)
from neurobench_age.pipelines.frozen_probe_training import (
    load_frozen_probe_training_manifest,
)
from neurobench_age.research.protocol import ProtocolError, load_study_protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layerwise-protocol", required=True, type=Path)
    parser.add_argument("--primary-protocol", required=True, type=Path)
    parser.add_argument("--layerwise-training-manifest", required=True, type=Path)
    parser.add_argument("--primary-training-manifest", required=True, type=Path)
    parser.add_argument("--layerwise-cache-root", required=True, type=Path)
    parser.add_argument("--primary-cache-root", required=True, type=Path)
    parser.add_argument("--output-cache-root", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        layerwise_protocol = load_study_protocol(
            args.layerwise_protocol, profile="layerwise"
        )
        primary_protocol = load_study_protocol(args.primary_protocol, profile="primary")
        if tuple(layerwise_protocol.encoder.layer_indices) != (-4, -3, -2, -1):
            raise FrozenEncoderError("layer-wise pooled cache requires layers -4 through -1")
        layerwise_records = load_frozen_probe_training_manifest(
            args.layerwise_training_manifest, protocol=layerwise_protocol
        )
        primary_records = load_frozen_probe_training_manifest(
            args.primary_training_manifest, protocol=primary_protocol
        )
        primary_by_subject = {record.subject_id: record for record in primary_records}
        if set(primary_by_subject) != {record.subject_id for record in layerwise_records}:
            raise FrozenEncoderError("primary and layer-wise training manifests have different subjects")
        args.output_cache_root.mkdir(parents=True, exist_ok=True)
        for layerwise_record in layerwise_records:
            primary_record = primary_by_subject[layerwise_record.subject_id]
            extra = load_cached_representations(
                args.layerwise_cache_root,
                layerwise_record.cache_identity,
                required_layers=(-4, -3),
            )
            primary = load_cached_representations(
                args.primary_cache_root,
                primary_record.cache_identity,
                required_layers=(-2, -1),
            )
            combined = {
                -4: extra[-4],
                -3: extra[-3],
                -2: primary[-2].mean(dim=1, keepdim=True).contiguous(),
                -1: primary[-1].mean(dim=1, keepdim=True).contiguous(),
            }
            first_shape = tuple(combined[-4].shape)
            if any(tuple(tensor.shape) != first_shape for tensor in combined.values()):
                raise FrozenEncoderError(
                    f"pooled layer shapes differ for subject {layerwise_record.subject_id}"
                )
            metadata = inspect_cached_representation_metadata(
                args.layerwise_cache_root,
                layerwise_record.cache_identity,
                expected_layers=(-4, -3),
            )
            evidence = {
                "encoder_frozen": True,
                "encoder_eval_mode": True,
                "inference_mode": True,
                "layer_indices": [-4, -3, -2, -1],
                "state_sha256_before": "0" * 64,
                "state_sha256_after": "0" * 64,
                "representation_transform": "arithmetic_mean_tokens",
                "source_cache_keys": {
                    "early": layerwise_record.cache_identity.key,
                    "final": primary_record.cache_identity.key,
                },
                "source_tensor_metadata": {
                    "early": {str(layer): metadata[layer].shape for layer in (-4, -3)},
                },
            }
            extra_metadata = json.loads(
                (args.layerwise_cache_root / layerwise_record.cache_identity.key / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            extra_evidence = extra_metadata.get("evidence", {})
            evidence["state_sha256_before"] = extra_evidence.get(
                "state_sha256_before", "0" * 64
            )
            evidence["state_sha256_after"] = extra_evidence.get(
                "state_sha256_after", "0" * 64
            )
            output_entry = args.output_cache_root / layerwise_record.cache_identity.key
            if output_entry.exists():
                loaded = load_cached_representations(
                    args.output_cache_root,
                    layerwise_record.cache_identity,
                    required_layers=(-4, -3, -2, -1),
                )
                if any(not torch.equal(loaded[layer], combined[layer]) for layer in combined):
                    raise FrozenEncoderError(
                        f"existing pooled cache differs for subject {layerwise_record.subject_id}"
                    )
            else:
                write_cached_representations(
                    args.output_cache_root,
                    layerwise_record.cache_identity,
                    combined,
                    evidence=evidence,
                    declared_layers=(-4, -3, -2, -1),
                )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "subject_count": len(layerwise_records),
                    "output_cache_root": str(args.output_cache_root.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ProtocolError, FrozenEncoderError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
