#!/usr/bin/env python3
"""Run the sealed ds006780 external transfer without aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from neurobench_age.pipelines.ds006780_external import (
    Ds006780ExternalError,
    run_ds006780_external,
)


def _device(value: str) -> str:
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target-free-manifest", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--qc-root", required=True, type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        inventory = run_ds006780_external(
            lock_path=args.lock,
            external_config_path=args.config,
            target_free_manifest_path=args.target_free_manifest,
            target_manifest_path=args.target_manifest,
            checkpoint_inventory_path=args.checkpoint_inventory,
            checkpoint_root=args.checkpoint_root,
            bids_root=args.bids_root,
            qc_root=args.qc_root,
            mapping_path=args.mapping,
            output_root=args.output_root,
            project_root=args.source_root,
            device=_device(args.device),
            extraction_batch_size=args.extraction_batch_size,
        )
    except (OSError, ValueError, Ds006780ExternalError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": inventory["status"],
                "prediction_count": inventory["prediction_count"],
                "prediction_inventory_sha256": inventory["prediction_inventory_sha256"],
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
