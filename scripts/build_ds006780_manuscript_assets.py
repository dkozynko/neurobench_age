#!/usr/bin/env python3
"""Render aggregate-only LaTeX and figure assets for ds006780 v5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.analysis.ds006780_manuscript_assets import build_assets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--precision-gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if any(
        not path.is_absolute()
        for path in (args.analysis, args.precision_gate, args.output_dir)
    ):
        parser.error("analysis, precision-gate, and output paths must be absolute")
    try:
        result = build_assets(
            analysis_path=args.analysis,
            precision_gate_path=args.precision_gate,
            output_dir=args.output_dir,
        )
    except Exception as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
