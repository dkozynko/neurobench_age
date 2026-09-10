"""Run target-free signal QC for every candidate in a ds006780 manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from neurobench_age.data.ds006780 import (
    Ds006780Error,
    build_target_free_qc,
    verify_ds006780_manifest,
    write_qc_report,
)
from neurobench_age.research.strict_json import load_json_strict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "research" / "ds006780_external_transfer.json"
QC_SCHEMA = ROOT / "schemas" / "research" / "ds006780_qc.schema.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-real-data", action="store_true")
    args = parser.parse_args()
    if not args.allow_real_data:
        raise SystemExit("full real-data QC requires --allow-real-data")

    manifest = load_json_strict(args.manifest)
    config = load_json_strict(args.config)
    verify_ds006780_manifest(
        args.bids_root,
        manifest,
        config_path=args.config,
        project_root=ROOT,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed = 0
    failures: list[dict[str, str]] = []
    for candidate in manifest["candidate_runs"]:
        subject_id = candidate["subject_id"]
        output_path = args.output_dir / f"{subject_id}_run-01_qc.json"
        try:
            windows, qc = build_target_free_qc(
                args.bids_root,
                candidate,
                manifest_sha256=manifest["manifest_sha256"],
                config=config,
            )
            if output_path.exists():
                existing = load_json_strict(output_path)
                if existing.get("qc_sha256") != qc.get("qc_sha256"):
                    raise Ds006780Error(f"existing QC artifact differs: {output_path}")
            else:
                write_qc_report(output_path, qc, schema_path=QC_SCHEMA)
            passed += 1
            print(json.dumps({"subject_id": subject_id, "status": "passed", "window_shape": list(windows.shape)}))
        except (Ds006780Error, OSError, ValueError) as error:
            failures.append(
                {
                    "subject_id": subject_id,
                    "run_id": candidate["run_id"],
                    "code": str(error).split(":", 1)[0],
                    "message": str(error),
                }
            )
            print(json.dumps({"subject_id": subject_id, "status": "failed", "error": str(error)}))

    summary = {
        "manifest_sha256": manifest["manifest_sha256"],
        "candidate_count": len(manifest["candidate_runs"]),
        "passed_count": passed,
        "failed_count": len(failures),
        "failure_codes": dict(Counter(item["code"] for item in failures)),
        "failures": failures,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
