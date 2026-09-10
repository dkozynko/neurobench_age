"""Join target-free signal QC to participant ages in a separate artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurobench_age.data.ds006780_cohort import (
    finalize_ds006780_cohort,
    write_target_manifest,
)
from neurobench_age.research.strict_json import load_json_strict


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "research" / "ds006780_target_manifest.schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-free-manifest", type=Path, required=True)
    parser.add_argument("--participant-metadata", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target_free = load_json_strict(args.target_free_manifest)
    qc_reports = {}
    for candidate in target_free["candidate_runs"]:
        subject_id = candidate["subject_id"]
        qc_path = args.qc_dir / f"{subject_id}_run-01_qc.json"
        qc_reports[subject_id] = load_json_strict(qc_path)
    manifest = finalize_ds006780_cohort(
        target_free,
        participant_metadata_path=args.participant_metadata,
        qc_reports=qc_reports,
    )
    written = write_target_manifest(args.output, manifest, schema_path=SCHEMA)
    print(written["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
