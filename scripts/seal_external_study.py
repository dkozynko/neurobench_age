#!/usr/bin/env python3
"""Seal the frozen-probe study after all manifests and checkpoints are ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from neurobench_age.research.protocol import load_study_protocol
from neurobench_age.research.study_lock import StudyLockError, seal_study


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        protocol = load_study_protocol(args.protocol)
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise StudyLockError("seal payload must contain a JSON object")
        if payload.get("protocol_sha256") != protocol.sha256:
            raise StudyLockError("seal payload protocol_sha256 does not match --protocol")
        lock = seal_study(args.lock, payload)
    except (OSError, json.JSONDecodeError, ValueError, StudyLockError) as error:
        parser.error(str(error))
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
