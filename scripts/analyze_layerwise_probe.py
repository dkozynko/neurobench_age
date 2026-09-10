#!/usr/bin/env python3
"""Analyze immutable external predictions from the layer-wise probe study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.analysis.layerwise import (
    LayerwiseAnalysisError,
    analyze_layerwise_artifacts,
    load_layerwise_analysis_config,
)
from neurobench_age.research.protocol import load_study_protocol
from neurobench_age.research.training_protocol import (
    load_frozen_probe_training_protocol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--training-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _write_exact_resume(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise LayerwiseAnalysisError(
                f"existing layer-wise analysis differs: {path}"
            )
        return
    path.write_bytes(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = (
        args.artifact_root,
        args.analysis_config,
        args.protocol,
        args.training_protocol,
        args.output,
    )
    if any(not path.is_absolute() for path in paths):
        parser.error("all analysis artifact, protocol, and output paths must be absolute")
    try:
        config = load_layerwise_analysis_config(args.analysis_config)
        protocol = load_study_protocol(args.protocol, profile="layerwise")
        training = load_frozen_probe_training_protocol(
            args.training_protocol, profile="layerwise"
        )
        result = analyze_layerwise_artifacts(
            artifact_root=args.artifact_root,
            config=config,
            expected_protocol_sha256=protocol.sha256,
            expected_training_protocol_sha256=training.sha256,
        )
        _write_exact_resume(args.output, result)
    except Exception as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": result["status"],
                "analysis_sha256": result["analysis_sha256"],
                "comparison_count": len(result["comparisons"]),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
