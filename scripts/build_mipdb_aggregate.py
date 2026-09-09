#!/usr/bin/env python3
"""Build or publish a disclosure-safe aggregate MIPDB supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurobench_age.analysis.mipdb_aggregate import (
    AggregateError,
    build_candidate,
    load_json_strict,
    load_participants_tsv,
    sha256_file,
    validate_completed_study_lock,
)
from neurobench_age.analysis.mipdb_aggregate_release import (
    ReleaseError,
    publish_candidate,
    validate_candidate,
    validate_ledger_for_candidate,
)


def _as_object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregateError(f"{label} must be a JSON object")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_output_path(
    path: Path,
    *,
    repository_root: Path,
    raw_data_root: Path | None,
    input_paths: Sequence[Path],
) -> Path:
    """Reject output paths that could overwrite code, data, or source inputs."""

    output = Path(path)
    if not output.is_absolute():
        raise AggregateError("output paths must be absolute")
    if output.exists() and output.is_symlink():
        raise AggregateError(f"output path must not be a symlink: {output}")
    resolved = output.resolve(strict=False)
    protected_roots = [Path(repository_root).resolve()]
    if raw_data_root is not None:
        protected_roots.append(Path(raw_data_root).resolve())
    if any(_is_within(resolved, root) for root in protected_roots):
        raise AggregateError("output path is inside a protected repository or data root")
    resolved_inputs = {Path(item).resolve(strict=False) for item in input_paths}
    if resolved in resolved_inputs:
        raise AggregateError("output path would overwrite an input artifact")
    if output.exists() and output.is_dir():
        raise AggregateError("output path is a directory")
    return output


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise AggregateError(f"candidate already exists: {path}") from error


def _verify_artifact_bindings(
    *,
    participants: Path,
    draft_manifest: Path,
    final_manifest: Path,
    cohort_qc: Path,
    source_manifest: Path,
    study_lock: Path,
    source_identity: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], tuple[dict[str, Any], ...]]:
    """Verify file-level and internal provenance before aggregation."""

    if sha256_file(source_manifest) != source_identity.get("source_manifest_sha256"):
        raise AggregateError("source manifest file does not match dataset identity")
    if sha256_file(final_manifest) != source_hashes.get("mipdb_manifest_sha256"):
        raise AggregateError("final MIPDB manifest hash does not match source hashes")
    if sha256_file(cohort_qc) != source_hashes.get("cohort_qc_sha256"):
        raise AggregateError("cohort QC file hash does not match source hashes")

    draft = _as_object(load_json_strict(draft_manifest), label="draft manifest")
    final = _as_object(load_json_strict(final_manifest), label="final manifest")
    qc = _as_object(load_json_strict(cohort_qc), label="cohort QC")
    rows = load_participants_tsv(participants)
    if final.get("draft_manifest_sha256") != sha256_file(draft_manifest):
        raise AggregateError("final manifest does not bind the draft manifest file")
    if qc.get("draft_manifest_sha256") != final.get("draft_manifest_sha256"):
        raise AggregateError("cohort QC does not bind the finalized draft identity")
    if qc.get("cohort_qc_sha256") != final.get("cohort_qc_sha256"):
        raise AggregateError("final manifest does not bind the cohort QC report")
    if qc.get("dataset_manifest_sha256") != final.get("dataset_manifest_sha256"):
        raise AggregateError("cohort QC does not bind the dataset identity")
    return draft, final, qc, rows


def _candidate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    input_paths = [
        args.participants,
        args.draft_manifest,
        args.final_manifest,
        args.cohort_qc,
        args.source_manifest_file,
        args.source_identity,
        args.source_hashes,
        args.study_lock,
    ]
    output = _safe_output_path(
        args.candidate_output,
        repository_root=args.repository_root,
        raw_data_root=args.raw_data_root,
        input_paths=input_paths,
    )
    source_identity = _as_object(
        load_json_strict(args.source_identity), label="source identity"
    )
    source_hashes = _as_object(
        load_json_strict(args.source_hashes), label="source hashes"
    )
    validate_completed_study_lock(args.study_lock, source_hashes=source_hashes)
    draft, final, qc, rows = _verify_artifact_bindings(
        participants=args.participants,
        draft_manifest=args.draft_manifest,
        final_manifest=args.final_manifest,
        cohort_qc=args.cohort_qc,
        source_manifest=args.source_manifest_file,
        study_lock=args.study_lock,
        source_identity=source_identity,
        source_hashes=source_hashes,
    )
    candidate = build_candidate(
        metadata_row_count=len(rows),
        metadata_rows=rows,
        draft_manifest=draft,
        final_manifest=final,
        cohort_qc=qc,
        dataset_identity=source_identity,
        source_hashes=source_hashes,
    )
    validate_candidate(candidate)
    _write_create_only(output, candidate)
    print(json.dumps({"candidate_digest_sha256": candidate["candidate_digest_sha256"], "output": str(output)}, sort_keys=True))
    return 0


def _publish(args: argparse.Namespace) -> int:
    input_paths = [args.candidate, args.approval, args.release_ledger]
    output = _safe_output_path(
        args.publish_output,
        repository_root=args.repository_root,
        raw_data_root=args.raw_data_root,
        input_paths=input_paths,
    )
    candidate = _as_object(load_json_strict(args.candidate), label="candidate")
    approval = _as_object(load_json_strict(args.approval), label="approval")
    ledger = _as_object(load_json_strict(args.release_ledger), label="release ledger")
    validate_ledger_for_candidate(candidate, ledger)
    publish_candidate(candidate, approval, output)
    print(json.dumps({"candidate_digest_sha256": candidate["candidate_digest_sha256"], "output": str(output)}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate = subparsers.add_parser("candidate", help="build a controlled aggregate candidate")
    candidate.add_argument("--participants", required=True, type=Path)
    candidate.add_argument("--draft-manifest", required=True, type=Path)
    candidate.add_argument("--final-manifest", required=True, type=Path)
    candidate.add_argument("--cohort-qc", required=True, type=Path)
    candidate.add_argument("--source-manifest-file", required=True, type=Path)
    candidate.add_argument("--source-identity", required=True, type=Path)
    candidate.add_argument("--source-hashes", required=True, type=Path)
    candidate.add_argument("--study-lock", required=True, type=Path)
    candidate.add_argument("--candidate-output", required=True, type=Path)
    candidate.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    candidate.add_argument("--raw-data-root", type=Path)

    publish = subparsers.add_parser("publish", help="publish an approved controlled aggregate")
    publish.add_argument("--candidate", required=True, type=Path)
    publish.add_argument("--approval", required=True, type=Path)
    publish.add_argument("--release-ledger", required=True, type=Path)
    publish.add_argument("--publish-output", required=True, type=Path)
    publish.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    publish.add_argument("--raw-data-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "candidate":
            return _candidate(args, parser)
        return _publish(args)
    except (AggregateError, ReleaseError, OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
