#!/usr/bin/env python3
"""Run the sealed capacity--data extension matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
from typing import Sequence

import torch

from neurobench_age.core.evidence import source_tree_sha256
from neurobench_age.pipelines.capacity_data_regime import (
    build_cache_manifest_from_records,
    compute_capacity_preflight,
    run_capacity_data_regime,
    run_capacity_pilot,
)
from neurobench_age.pipelines.frozen_probe_training import (
    _available_memory_bytes,
    load_frozen_probe_training_manifest,
)
from neurobench_age.research.capacity_data_regime import (
    build_nested_training_cohorts,
    load_capacity_data_regime_protocol,
)
from neurobench_age.research.capacity_data_regime_lock import build_lock_core
from neurobench_age.research.protocol import load_study_protocol
from neurobench_age.research.training_protocol import load_frozen_probe_training_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--primary-study-lock", required=True, type=Path)
    parser.add_argument("--primary-prediction-inventory", required=True, type=Path)
    parser.add_argument("--estimated-extension-output-bytes", required=True, type=int)
    parser.add_argument("--pilot-seconds", type=float)
    parser.add_argument("--raw-data-root", action="append", type=Path, default=[])
    parser.add_argument("--primary-evidence-root", action="append", type=Path, default=[])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate extension inputs and storage without starting training",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="run the declared n=200/mean_linear/seed=33 pilot and stop",
    )
    return parser


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cohort_rows(path: Path) -> list[dict[str, object]]:
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"release", "subject", "split"}
            if set(reader.fieldnames or ()) < required:
                raise ValueError("cohort manifest must contain release, subject, split")
            rows = [
                {
                    "release": row["release"],
                    "subject_id": row["subject"],
                    "split": "validation" if row["split"] == "val" else row["split"],
                }
                for row in reader
            ]
    except (OSError, csv.Error, KeyError, ValueError) as error:
        raise ValueError(f"could not read cohort manifest: {path}") from error
    if not rows:
        raise ValueError("cohort manifest is empty")
    return rows


def _runtime_hashes(device: str) -> tuple[str, str]:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device_request": device,
    }
    hardware = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    return _canonical_sha256(environment), _canonical_sha256(hardware)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    try:
        extension_protocol = load_capacity_data_regime_protocol(
            args.protocol, repository_root=repository_root
        )
        representation_protocol_path = (
            repository_root / extension_protocol.representation_protocol_path
        )
        representation_protocol = load_study_protocol(representation_protocol_path)
        training_protocol = load_frozen_probe_training_protocol(
            repository_root / extension_protocol.training_protocol_path
        )
        records = load_frozen_probe_training_manifest(
            args.training_manifest, protocol=representation_protocol
        )
        cohort_rows = _load_cohort_rows(args.cohort_manifest)
        cohorts = build_nested_training_cohorts(cohort_rows, protocol=extension_protocol)
        cache_manifest, cache_digest, cache_bytes, cached_windows, train_windows = (
            build_cache_manifest_from_records(records, cache_root=args.cache_root)
        )
        del cache_manifest
        available_memory_bytes = _available_memory_bytes()
        resolved_device = (
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        if args.pilot_only and args.preflight_only:
            parser.error("--pilot-only and --preflight-only cannot be combined")
        if args.pilot_only:
            pilot_root = Path(f"{Path(args.output_root).resolve()}.pilot")
            pilot = run_capacity_pilot(
                protocol=extension_protocol,
                training=training_protocol,
                records=records,
                cohorts=cohorts,
                cache_root=args.cache_root,
                output_root=pilot_root,
                device=resolved_device,
                training_source_sha256=source_tree_sha256(repository_root),
                available_memory_bytes=available_memory_bytes,
            )
            print(json.dumps(pilot, sort_keys=True, allow_nan=False))
            return 0
        free_space_bytes = shutil.disk_usage(Path(args.output_root).parent).free
        pilot_seconds = 0.0 if args.pilot_seconds is None else args.pilot_seconds
        preflight = compute_capacity_preflight(
            representation_cache_bytes=cache_bytes,
            estimated_extension_output_bytes=args.estimated_extension_output_bytes,
            free_space_bytes=free_space_bytes,
            available_memory_bytes=available_memory_bytes,
            projected_store_bytes=cache_bytes,
            train_windows=train_windows,
            pilot_seconds=pilot_seconds,
            cached_window_count=cached_windows,
            free_space_floor_bytes=extension_protocol.free_space_floor_bytes,
            free_space_cache_fraction=extension_protocol.free_space_cache_fraction,
            free_space_output_multiplier=extension_protocol.free_space_output_multiplier,
        )
        validation_subjects = sorted(
            (record.subject_id for record in records if record.split == "validation"),
            key=lambda subject_id: subject_id.encode("utf-8"),
        )
        primary_hashes = {
            "parent_primary_study_lock_sha256": _sha256_file(args.primary_study_lock),
            "parent_primary_prediction_inventory_sha256": _sha256_file(
                args.primary_prediction_inventory
            ),
        }
        environment_sha256, hardware_sha256 = _runtime_hashes(args.device)
        core = {
            "extension_id": extension_protocol.extension_id,
            "schema_version": 1,
            **primary_hashes,
            "extension_protocol_sha256": extension_protocol.sha256,
            "representation_protocol_sha256": representation_protocol.sha256,
            "training_protocol_sha256": training_protocol.sha256,
            "training_source_sha256": source_tree_sha256(repository_root),
            "environment_sha256": environment_sha256,
            "hardware_sha256": hardware_sha256,
            "hbn_manifest_sha256": _sha256_file(args.cohort_manifest),
            "hbn_training_manifest_sha256": _sha256_file(args.training_manifest),
            "representation_cache_manifest_sha256": cache_digest,
            "validation_subject_list_sha256": _canonical_sha256(validation_subjects),
            "cohort_hashes": {
                f"train_{size}": _canonical_sha256(
                    {
                        "extension_version": extension_protocol.extension_id,
                        "split_name": f"train-{size}",
                        "subject_ids": list(cohorts[size]),
                        "source_manifest_sha256": _sha256_file(args.cohort_manifest),
                        "representation_cache_sha256": cache_digest,
                    }
                )
                for size in extension_protocol.training_sizes
            },
            "expected_run_count": extension_protocol.expected_run_count,
            "expected_prediction_count": extension_protocol.expected_prediction_count,
            "output_root_identity": _canonical_sha256(
                {"path": str(Path(args.output_root).resolve())}
            ),
            "preflight": preflight,
        }
        core_bundle = build_lock_core(core)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "protocol_sha256": extension_protocol.sha256,
                        "cache_manifest_sha256": cache_digest,
                        "lock_core_sha256": core_bundle["lock_core_sha256"],
                        "preflight": preflight,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.pilot_seconds is None or args.pilot_seconds <= 0:
            parser.error("full extension run requires --pilot-seconds from a completed one-run pilot")
        result = run_capacity_data_regime(
            protocol=extension_protocol,
            training=training_protocol,
            records=records,
            cohorts=cohorts,
            cache_root=args.cache_root,
            output_root=args.output_root,
            repository_root=repository_root,
            raw_data_roots=tuple(args.raw_data_root),
            primary_evidence_roots=tuple(args.primary_evidence_root),
            primary_study_lock=args.primary_study_lock,
            primary_prediction_inventory=args.primary_prediction_inventory,
            training_source_sha256=core["training_source_sha256"],
            core_bundle=core_bundle,
            preflight_report=preflight,
            device=resolved_device,
            available_memory_bytes=available_memory_bytes,
            progress_sink=lambda event: print(
                json.dumps(event, sort_keys=True, allow_nan=False), flush=True
            ),
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "run_count": result["run_count"],
                    "checkpoint_inventory_body_sha256": result[
                        "checkpoint_inventory"
                    ]["checkpoint_inventory_body_sha256"],
                    "checkpoint_sealed_lock_sha256": result[
                        "checkpoint_sealed_lock"
                    ]["lock_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
