"""Build or verify a target-free ds006780 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurobench_age.data.ds006780 import (
    build_target_free_manifest,
    verify_ds006780_manifest,
    write_manifest,
)
from neurobench_age.research.strict_json import load_json_strict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "research" / "ds006780_external_transfer.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "research" / "ds006780_target_free_manifest.schema.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--source-commit")
    parser.add_argument("--openneuro-version")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_json_strict(args.config)
    dataset = config["dataset"]
    source_commit = args.source_commit or dataset["source_commit"]
    openneuro_version = args.openneuro_version or dataset["openneuro_version"]
    if args.verify_manifest is not None:
        if args.output is not None:
            raise SystemExit("--verify-manifest cannot be combined with --output")
        manifest = load_json_strict(args.verify_manifest)
        verify_ds006780_manifest(
            args.bids_root,
            manifest,
            config_path=args.config,
            project_root=ROOT,
        )
        print(f"verified {args.verify_manifest}")
        return 0
    if args.output is None:
        raise SystemExit("--output is required when building a manifest")
    manifest = build_target_free_manifest(
        args.bids_root,
        config_path=args.config,
        project_root=ROOT,
        source_commit=source_commit,
        openneuro_version=openneuro_version,
    )
    written = write_manifest(args.output, manifest, schema_path=args.schema)
    print(written["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
