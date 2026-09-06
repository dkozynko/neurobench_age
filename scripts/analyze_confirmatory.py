#!/usr/bin/env python3
"""Run the sealed confirmatory analysis over completed external predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.analysis.confirmatory import (
    ConfirmatoryAnalysisError,
    analyze_confirmatory_study,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--mipdb-manifest", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        artifact_paths = (
            args.lock,
            args.checkpoint_root,
            args.checkpoint_inventory,
            args.mipdb_manifest,
            args.prediction_root,
            args.output_root,
        )
        if any(not path.is_absolute() for path in artifact_paths):
            raise ConfirmatoryAnalysisError(
                "lock, artifact, prediction, and output paths must be absolute"
            )
        report = analyze_confirmatory_study(
            protocol_path=args.protocol,
            lock_path=args.lock,
            checkpoint_root=args.checkpoint_root,
            checkpoint_inventory_path=args.checkpoint_inventory,
            mipdb_manifest_path=args.mipdb_manifest,
            prediction_root=args.prediction_root,
            analysis_output_root=args.output_root,
        )
    except (OSError, ConfirmatoryAnalysisError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "status": report["status"],
                "analysis_sha256": report["analysis_sha256"],
                "established_heads": report["established_heads"],
                "conclusion": report["conclusion"],
                "output": str(
                    (args.output_root / "confirmatory_analysis.json").resolve()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
