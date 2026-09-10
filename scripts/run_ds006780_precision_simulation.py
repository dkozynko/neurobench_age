"""Run a planning-only precision simulation from a verified target-free manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurobench_age.data.ds006780 import verify_ds006780_manifest
from neurobench_age.research.precision_simulation import (
    PrecisionSimulationError,
    scenario_from_mapping,
    simulate_precision_scenarios,
)
from neurobench_age.research.strict_json import (
    canonical_sha256,
    load_json_strict,
    reject_target_fields,
    validate_schema,
    write_create_only_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "research" / "ds006780_external_transfer.json"
SCENARIO_SCHEMA = ROOT / "schemas" / "research" / "ds006780_precision_scenarios.schema.json"
OUTPUT_SCHEMA = ROOT / "schemas" / "research" / "ds006780_precision_simulation.schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulation-replicates", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--allow-real-data", action="store_true")
    args = parser.parse_args()
    if not args.allow_real_data:
        raise SystemExit("precision simulation against a real manifest requires --allow-real-data")

    manifest = load_json_strict(args.manifest)
    config = load_json_strict(args.config)
    verify_ds006780_manifest(
        args.bids_root,
        manifest,
        config_path=args.config,
        project_root=ROOT,
    )
    scenario_payload = load_json_strict(args.scenarios)
    validate_schema(scenario_payload, SCENARIO_SCHEMA)
    scenarios = [scenario_from_mapping(item) for item in scenario_payload["scenarios"]]
    results = simulate_precision_scenarios(
        scenarios,
        subject_count=len(manifest["candidate_runs"]),
        simulation_replicates=args.simulation_replicates,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    body = {
        "schema_version": 1,
        "status": "planning_only",
        "study_id": str(scenario_payload["study_id"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": canonical_sha256(config),
        "scenario_source_sha256": canonical_sha256(scenario_payload),
        "subject_count": len(manifest["candidate_runs"]),
        "seed_ids": list(range(33, 43)),
        "simulation_replicates": args.simulation_replicates,
        "bootstrap_iterations": args.bootstrap_iterations,
        "scenarios": results,
    }
    reject_target_fields(body)
    written = write_create_only_json(
        args.output,
        body,
        schema_path=OUTPUT_SCHEMA,
        hash_field="simulation_sha256",
    )
    print(
        f"wrote planning-only precision simulation: {args.output} "
        f"({len(written['scenarios'])} scenarios, n={written['subject_count']})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrecisionSimulationError as error:
        raise SystemExit(str(error)) from error
