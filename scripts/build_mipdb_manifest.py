#!/usr/bin/env python3
"""Build the model-free, content-addressed MIPDB draft cohort manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from neurobench_age.data.medium_subset import manifest_sha256, read_manifest
from neurobench_age.data.mipdb import MipdbInventoryError, build_mipdb_inventory
from neurobench_age.research.protocol import ProtocolError, load_study_protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--hbn-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        protocol = load_study_protocol(args.protocol)
        hbn_rows = read_manifest(args.hbn_manifest)
        training_rows = [row for row in hbn_rows if row.split == "train"]
        training_subjects = [row.subject for row in training_rows]
        if not training_rows:
            raise MipdbInventoryError("HBN manifest has no training subjects")
        if len(training_subjects) != len(set(training_subjects)):
            raise MipdbInventoryError("HBN training subjects are not unique")
        training_ages = [float(row.age) for row in training_rows]
        if not all(math.isfinite(age) for age in training_ages):
            raise MipdbInventoryError("HBN training ages must be finite")
        manifest = build_mipdb_inventory(
            args.bids_root,
            protocol=protocol,
            hbn_age_support=(min(training_ages), max(training_ages)),
            hbn_age_support_manifest_sha256=manifest_sha256(args.hbn_manifest),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, ProtocolError, MipdbInventoryError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
                "pilot_subjects": len(manifest["cohorts"]["pilot"]),
                "primary_subjects": len(manifest["cohorts"]["primary"]),
                "underpowered": manifest["underpowered"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
