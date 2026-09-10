#!/usr/bin/env python3
"""Render aggregate-only assets for the layer-wise exploratory analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.analysis.layerwise_assets import build_layerwise_assets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = (args.analysis, args.output_root)
    if any(not path.is_absolute() for path in paths):
        parser.error("analysis and output paths must be absolute")
    try:
        manifest = build_layerwise_assets(
            analysis_path=args.analysis,
            output_root=args.output_root,
            repository_root=Path(__file__).resolve().parents[1],
        )
    except Exception as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "asset_manifest_sha256": manifest["asset_manifest_sha256"],
                "asset_count": len(manifest["files"]),
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
