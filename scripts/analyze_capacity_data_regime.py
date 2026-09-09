#!/usr/bin/env python3
"""Analyze finalized capacity--data extension predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from neurobench_age.analysis.capacity_data_regime import analyze_capacity_data_regime
from neurobench_age.research.capacity_data_regime import load_capacity_data_regime_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--final-lock", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--prediction-inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    try:
        protocol = load_capacity_data_regime_protocol(
            args.protocol, repository_root=repository_root
        )
        result = analyze_capacity_data_regime(
            final_lock=_load_json(args.final_lock, "final extension lock"),
            prediction_inventory=_load_json(
                args.prediction_inventory, "prediction inventory"
            ),
            checkpoint_inventory=_load_json(
                args.checkpoint_inventory, "checkpoint inventory"
            ),
            protocol=protocol,
        )
        output = Path(args.output).resolve()
        forbidden_roots = (
            repository_root / "manuscript/generated",
            repository_root / "results/canonical",
        )
        if any(output == root.resolve() or output.is_relative_to(root.resolve()) for root in forbidden_roots):
            raise ValueError("extension analysis output cannot overwrite primary evidence or manuscript assets")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            existing = _load_json(output, "existing analysis")
            if existing != result:
                raise ValueError("existing analysis output conflicts with finalized evidence")
        else:
            output.write_text(
                json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "analysis_sha256": result["analysis_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        build_parser().error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
