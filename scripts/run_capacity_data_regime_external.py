#!/usr/bin/env python3
"""Run the lock-gated secondary MIPDB evaluation for the capacity extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import torch

from neurobench_age.pipelines.capacity_data_regime_external import (
    run_capacity_data_regime_external,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-sealed-lock", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--primary-study-lock", required=True, type=Path)
    parser.add_argument("--primary-prediction-inventory", required=True, type=Path)
    parser.add_argument(
        "--mipdb-subjects",
        required=True,
        type=Path,
        help="JSON array containing exactly 75 {subject_id, true_age, split} records",
    )
    parser.add_argument(
        "--representation-root",
        required=True,
        type=Path,
        help="directory containing one <subject_id>.pt tensor or {-2, -1} mapping per subject",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def _load_subjects(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read MIPDB subjects: {path}") from error
    if not isinstance(payload, list):
        raise ValueError("MIPDB subjects must be a JSON array")
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError("MIPDB subjects must contain only JSON objects")
    return [dict(item) for item in payload]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        subjects = _load_subjects(args.mipdb_subjects)

        def provider(subject_id: str) -> object:
            path = args.representation_root / f"{subject_id}.pt"
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except Exception as error:
                raise ValueError(f"could not load MIPDB representation: {path}") from error

        result = run_capacity_data_regime_external(
            checkpoint_sealed_lock_path=args.checkpoint_sealed_lock,
            checkpoint_inventory_path=args.checkpoint_inventory,
            checkpoint_root=args.checkpoint_root,
            primary_study_lock=args.primary_study_lock,
            primary_prediction_inventory=args.primary_prediction_inventory,
            subjects=subjects,
            output_root=args.output_root,
            representation_provider=provider,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "prediction_inventory_body_sha256": result["prediction_inventory"][
                        "prediction_inventory_body_sha256"
                    ],
                    "final_lock_sha256": result["final_lock"]["lock_sha256"],
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
