#!/usr/bin/env python3
"""Seal the ds006780 external transfer after all pre-inference gates pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.research.ds006780_external_lock import (
    Ds006780ExternalLockError,
    build_ds006780_external_lock,
    select_ds006780_runs,
)
from neurobench_age.research.strict_json import canonical_json_bytes, load_json_strict


def _write_create_only(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target-free-manifest", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-inventory", required=True, type=Path)
    parser.add_argument("--checkpoint-lock", required=True, type=Path)
    parser.add_argument("--encoder-checkpoint-sha256", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        config = load_json_strict(args.config)
        target_free = load_json_strict(args.target_free_manifest)
        target = load_json_strict(args.target_manifest)
        inventory = load_json_strict(args.checkpoint_inventory)
        checkpoint_lock = load_json_strict(args.checkpoint_lock)
        if not all(isinstance(value, dict) for value in (config, target_free, target, inventory, checkpoint_lock)):
            raise Ds006780ExternalLockError("all sealing inputs must be JSON objects")
        checkpoint_core = checkpoint_lock.get("lock_core")
        if not isinstance(checkpoint_core, dict):
            raise Ds006780ExternalLockError("checkpoint lock does not expose lock_core")
        selected = select_ds006780_runs(inventory)
        lock = build_ds006780_external_lock(
            external_config=config,
            target_free_manifest=target_free,
            target_manifest=target,
            checkpoint_inventory=inventory,
            selected_runs=selected,
            checkpoint_core=checkpoint_core,
            execution_source_sha256=source_tree_sha256(args.source_root),
            encoder_checkpoint_sha256=args.encoder_checkpoint_sha256,
            output_root=args.output_root,
        )
        _write_create_only(args.lock, lock)
    except (OSError, ValueError, Ds006780ExternalLockError) as error:
        parser.error(str(error))
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
