#!/usr/bin/env python3
"""Render aggregate-only tables and figures for the capacity--data extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.analysis.capacity_data_regime_assets import (
    CapacityDataRegimeAssetError,
    build_capacity_data_regime_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--final-lock", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--prediction-inventory", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_capacity_data_regime_assets(
            analysis_path=args.analysis,
            final_lock_path=args.final_lock,
            checkpoint_inventory_path=args.checkpoint_inventory,
            prediction_inventory_path=args.prediction_inventory,
            output_root=args.output_root,
            repository_root=Path(__file__).resolve().parents[1],
        )
    except CapacityDataRegimeAssetError as error:
        build_parser().error(str(error))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "asset_manifest_sha256": manifest["asset_manifest_sha256"],
                "asset_count": len(manifest["files"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
