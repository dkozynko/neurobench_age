"""Analyze the sealed ds006780 external prediction matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurobench_age.analysis.ds006780_external import (
    analyze_external_series,
    load_external_prediction_series,
)
from neurobench_age.research.ds006780_external_lock import load_ds006780_external_lock
from neurobench_age.research.strict_json import canonical_json_bytes, load_json_strict


def _write_create_only(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.write(b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()

    lock = load_ds006780_external_lock(args.lock)
    inventory = load_json_strict(args.inventory)
    if not isinstance(inventory, dict):
        raise SystemExit("prediction inventory must be a JSON object")
    series = load_external_prediction_series(
        lock=lock,
        inventory=inventory,
        output_root=args.output_root,
    )
    analysis = analyze_external_series(
        series,
        lock_sha256=str(lock["lock_sha256"]),
        prediction_inventory_sha256=str(inventory["prediction_inventory_sha256"]),
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=20260910,
        confidence=float(lock["precision_gate"]["confidence"]),
    )
    _write_create_only(args.output, analysis)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "analysis_sha256": analysis["analysis_sha256"],
                "primary_delta": analysis["primary_contrast"]["paired"]["mean_pearson_delta"],
                "primary_ci": analysis["primary_contrast"]["bootstrap"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

