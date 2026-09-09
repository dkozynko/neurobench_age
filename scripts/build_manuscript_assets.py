#!/usr/bin/env python3
"""Validate retained evidence and build deterministic manuscript assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurobench_age.analysis.manuscript_assets import (  # noqa: E402
    ManuscriptEvidenceError,
    export_compact_evidence,
    render_assets,
    validate_source_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import", help="validate and retain compact evidence")
    importer.add_argument("--analysis", type=Path, required=True)
    importer.add_argument("--lock", type=Path, required=True)
    importer.add_argument("--checkpoint-inventory", type=Path, required=True)
    importer.add_argument("--prediction-inventory", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    renderer = subparsers.add_parser("render", help="render deterministic derived assets")
    renderer.add_argument("--evidence", type=Path, required=True)
    renderer.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "import":
            bundle = validate_source_bundle(
                analysis_path=arguments.analysis,
                lock_path=arguments.lock,
                checkpoint_inventory_path=arguments.checkpoint_inventory,
                prediction_inventory_path=arguments.prediction_inventory,
            )
            manifest = export_compact_evidence(bundle, arguments.output)
            print(
                "status=complete "
                f"runs={len(bundle.run_pairs)} "
                f"subjects={len(bundle.subject_ids)} "
                f"predictions={bundle.prediction_count} "
                f"manifest_sha256={manifest['files']['study_summary.json']['sha256']}"
            )
            return 0
        manifest = render_assets(arguments.evidence, arguments.output)
        manifest_sha256 = (arguments.output / "assets_manifest.sha256").read_text(
            encoding="ascii"
        ).split()[0]
        print(
            "status=complete "
            f"assets={len(manifest['files'])} "
            f"assets_manifest_sha256={manifest_sha256}"
        )
        return 0
    except ManuscriptEvidenceError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
