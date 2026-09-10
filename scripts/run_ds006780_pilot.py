"""Run an opt-in target-free ds006780 signal and optional REVE pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neurobench_age.data.ds006780 import (
    build_target_free_qc,
    verify_ds006780_manifest,
    write_qc_report,
)
from neurobench_age.research.strict_json import load_json_strict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "research" / "ds006780_external_transfer.json"
QC_SCHEMA = ROOT / "schemas" / "research" / "ds006780_qc.schema.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--allow-real-data", action="store_true")
    parser.add_argument("--run-reve", action="store_true")
    parser.add_argument("--mapping-path", type=Path)
    return parser


def _run_reve_pilot(windows: np.ndarray, *, channel_names: list[str], mapping_path: Path | None) -> dict[str, object]:
    import torch

    from neurobench_age.pipelines.frozen_probe import (
        extract_frozen_representations,
        load_reve_encoder,
    )

    encoder = load_reve_encoder(
        "brain-bzh/reve-base",
        channel_names=channel_names,
        mapping_path=mapping_path,
        initialization_seed=0,
    )
    representations, evidence = extract_frozen_representations(
        encoder,
        torch.from_numpy(np.asarray(windows, dtype=np.float32)),
        layer_indices=(-2, -1),
    )
    return {
        "input_shape": list(windows.shape),
        "layer_shapes": {str(index): list(value.shape) for index, value in representations.items()},
        "evidence": evidence,
    }


def main() -> int:
    args = _parser().parse_args()
    if not args.allow_real_data:
        raise SystemExit("real-data pilot requires --allow-real-data")
    manifest = load_json_strict(args.manifest)
    config = load_json_strict(args.config)
    verify_ds006780_manifest(
        args.bids_root,
        manifest,
        config_path=args.config,
        project_root=ROOT,
    )
    candidates = [
        candidate
        for candidate in manifest["candidate_runs"]
        if candidate["subject_id"] == args.subject_id
    ]
    if len(candidates) != 1:
        raise SystemExit(f"expected one manifest candidate for {args.subject_id}")
    candidate = candidates[0]
    windows, qc = build_target_free_qc(
        args.bids_root,
        candidate,
        manifest_sha256=manifest["manifest_sha256"],
        config=config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc_path = args.output_dir / f"{args.subject_id}_run-01_qc.json"
    write_qc_report(qc_path, qc, schema_path=QC_SCHEMA)
    report: dict[str, object] = {
        "subject_id": args.subject_id,
        "run_id": "run-01",
        "qc_path": str(qc_path),
        "window_shape": list(windows.shape),
    }
    if args.run_reve:
        report["reve"] = _run_reve_pilot(
            windows,
            channel_names=qc["channel_labels"],
            mapping_path=args.mapping_path,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
