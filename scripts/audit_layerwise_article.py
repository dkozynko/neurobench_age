#!/usr/bin/env python3
"""Audit the retained layer-wise evidence and its manuscript integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neurobench_age.analysis.layerwise import _canonical_sha256  # noqa: E402


class LayerwiseArticleAuditError(ValueError):
    """Raised when retained evidence and manuscript claims are inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerwiseArticleAuditError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LayerwiseArticleAuditError(f"could not read JSON: {path}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise LayerwiseArticleAuditError(f"could not hash file: {path}") from error
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise LayerwiseArticleAuditError(f"could not read text file: {path}") from error


def _check_self_hash(payload: dict[str, Any], field: str, description: str) -> str:
    value = payload.get(field)
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{description} has an invalid {field}")
    body = dict(payload)
    body.pop(field)
    _require(_canonical_sha256(body) == value, f"{description} self-hash mismatch")
    return value


def audit_layerwise_article(repository_root: Path) -> dict[str, Any]:
    """Validate aggregate evidence, generated assets, and bounded article claims."""

    root = Path(repository_root).resolve()
    extension = root / "results/extensions/layerwise_probe_20260910"
    assets = extension / "assets"
    report = _load_json(extension / "layerwise_analysis.json")
    analysis_sha256 = _check_self_hash(report, "analysis_sha256", "analysis report")

    _require(report.get("status") == "complete", "layer-wise analysis is not complete")
    _require(report.get("scope") == "exploratory_secondary", "layer-wise scope is not exploratory")
    _require(report.get("baseline_head") == "mean_linear_layer_m1", "unexpected layer-wise baseline")
    _require(report.get("subject_count") == 75, "unexpected layer-wise subject count")
    seeds = report.get("seeds")
    _require(seeds == list(range(33, 43)), "unexpected layer-wise seed set")
    comparison_order = report.get("comparison_order")
    expected_order = [
        "mean_linear_layer_m4",
        "mean_linear_layer_m3",
        "mean_linear_layer_m2",
    ]
    _require(comparison_order == expected_order, "unexpected layer-wise comparison order")

    comparisons = report.get("comparisons")
    _require(isinstance(comparisons, dict), "layer-wise comparisons are missing")
    for head in expected_order:
        comparison = comparisons.get(head)
        _require(isinstance(comparison, dict), f"missing layer-wise comparison: {head}")
        paired = comparison.get("paired")
        bootstrap = comparison.get("bootstrap")
        _require(isinstance(paired, dict), f"missing paired statistics: {head}")
        _require(isinstance(bootstrap, dict), f"missing bootstrap statistics: {head}")
        _require(bootstrap.get("iterations") == 10000, f"unexpected bootstrap count: {head}")
        _require(bootstrap.get("subject_count") == 75, f"unexpected bootstrap cohort: {head}")
        _require(bootstrap.get("ci_low") <= 0 <= bootstrap.get("ci_high"),
                 f"layer-wise interval does not cross zero: {head}")
        _require(paired.get("seed_count") == 10 or len(paired.get("per_seed", [])) == 10,
                 f"unexpected seed count: {head}")

    manifest = _load_json(assets / "assets_manifest.json")
    manifest_sha256 = _check_self_hash(manifest, "asset_manifest_sha256", "asset manifest")
    _require(manifest.get("analysis_sha256") == analysis_sha256,
             "asset manifest is not bound to the analysis report")
    expected_files = {
        "assets_manifest.json",
        "layerwise_depth.pdf",
        "layerwise_results_macros.tex",
        "layerwise_seed_deltas.pdf",
        "layerwise_summary.tex",
    }
    _require(set(manifest.get("files", [])) == expected_files,
             "asset manifest file inventory is incomplete")
    file_hashes = manifest.get("file_hashes")
    _require(isinstance(file_hashes, dict), "asset manifest file hashes are missing")
    for name in expected_files - {"assets_manifest.json"}:
        path = assets / name
        _require(path.is_file(), f"missing retained layer-wise asset: {name}")
        expected = file_hashes.get(name)
        _require(isinstance(expected, dict), f"missing asset hash: {name}")
        _require(expected.get("bytes") == path.stat().st_size, f"asset size mismatch: {name}")
        _require(expected.get("sha256") == _sha256_file(path), f"asset hash mismatch: {name}")

    macros = _read_text(assets / "layerwise_results_macros.tex")
    suffixes = {
        "mean_linear_layer_m4": "MFour",
        "mean_linear_layer_m3": "MThree",
        "mean_linear_layer_m2": "MTwo",
    }
    for head, suffix in suffixes.items():
        delta = comparisons[head]["paired"]["mean_pearson_delta"]
        expected_macro = f"\\newcommand{{\\Layerwise{suffix}Delta}}{{{delta:.3f}}}"
        _require(expected_macro in macros, f"manuscript macro mismatch: {head}")
    _require(r"\newcommand{\LayerwiseSubjectCount}{75}" in macros,
             "manuscript subject-count macro mismatch")

    results = _read_text(root / "manuscript/sections/results.tex")
    _require(r"\LayerwiseSubjectCount{}" in results, "results section duplicates subject count")
    _require("crossed zero" in results, "results section omits interval boundary")
    _require("primary superiority claim" in results, "results section omits exploratory boundary")
    _require("stable improvement" not in results.casefold(),
             "results section promotes exploratory result")

    public_docs = "\n".join(
        _read_text(path)
        for path in (
            root / "README.md",
            root / "ARTICLE_SCOPE.md",
            root / "docs/research/article_evidence_registry.md",
            root / "docs/research/readiness_audit.md",
        )
    )
    _require(analysis_sha256 in public_docs, "analysis hash is absent from public audit docs")
    _require(manifest_sha256 in public_docs, "asset-manifest hash is absent from public audit docs")

    aggregate_text = "\n".join(
        _read_text(path)
        for path in extension.rglob("*")
        if path.is_file() and path.suffix in {".json", ".md", ".tex"}
    )
    _require('"subject_id"' not in aggregate_text, "participant identifier leaked into aggregate evidence")
    _require("/" + "workspace/" not in aggregate_text, "remote path leaked into aggregate evidence")
    _require("/" + "Users/" not in aggregate_text, "local path leaked into aggregate evidence")
    _require(re.search(r"hf_[A-Za-z0-9]{20,}", aggregate_text) is None,
             "credential leaked into aggregate evidence")

    return {
        "status": "pass",
        "analysis_sha256": analysis_sha256,
        "asset_manifest_sha256": manifest_sha256,
        "comparison_count": len(expected_order),
        "scope": report["scope"],
        "seed_count": len(seeds),
        "subject_count": report["subject_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPO_ROOT,
        help="repository containing retained evidence and manuscript sources",
    )
    args = parser.parse_args(argv)
    try:
        result = audit_layerwise_article(args.repository_root)
    except LayerwiseArticleAuditError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
