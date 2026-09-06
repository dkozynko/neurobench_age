#!/usr/bin/env python3
"""Finalize MIPDB evaluation cohorts using target-free, predeclared signal QC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.data.mipdb import (
    MipdbInventoryError,
    MipdbPreprocessingError,
    finalize_mipdb_cohort,
)
from neurobench_age.research.protocol import ProtocolError, load_study_protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--draft-manifest", required=True, type=Path)
    parser.add_argument("--qc-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        paths = (
            args.bids_root,
            args.draft_manifest,
            args.qc_output,
            args.output,
        )
        if any(not path.is_absolute() for path in paths):
            raise MipdbInventoryError("all data and artifact paths must be absolute")
        finalized = finalize_mipdb_cohort(
            bids_root=args.bids_root,
            draft_manifest_path=args.draft_manifest,
            protocol=load_study_protocol(args.protocol),
            qc_output_path=args.qc_output,
            output_path=args.output,
        )
    except (
        OSError,
        ProtocolError,
        MipdbInventoryError,
        MipdbPreprocessingError,
        ValueError,
    ) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "status": finalized["status"],
                "primary_subjects": len(finalized["cohorts"]["primary"]),
                "extrapolation_subjects": len(
                    finalized["cohorts"]["extrapolation"]
                ),
                "underpowered": finalized["underpowered"],
                "cohort_qc_sha256": finalized["cohort_qc_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
