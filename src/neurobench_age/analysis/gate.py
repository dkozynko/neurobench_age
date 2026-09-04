"""Validation-only finalist gate for the article protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


class GateError(ValueError):
    """Raised when validation evidence cannot be used for a finalist gate."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"could not read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def _validation_score(run_dir: Path) -> tuple[int, float]:
    manifest = _read_object(run_dir / "run_manifest.json")
    selection = _read_object(run_dir / "selection.json")
    report = _read_object(run_dir / "report.json")

    if manifest.get("status") != "completed":
        raise GateError(f"incomplete validation run: {run_dir}")
    if manifest.get("evaluation_mode") != "validation_only":
        raise GateError(f"validation-only evidence required: {run_dir}")
    if manifest.get("test_access") != "sealed":
        raise GateError(f"test access is not sealed: {run_dir}")
    if report.get("test_status") != "withheld":
        raise GateError(f"validation-only report is contaminated: {run_dir}")

    forbidden = (
        run_dir / "epoch_test_metrics.jsonl",
        run_dir / "predictions" / "test.jsonl",
    )
    if any(path.exists() for path in forbidden) or any(run_dir.glob("test_*.json")):
        raise GateError(f"validation-only run contains test evidence: {run_dir}")

    seed = manifest.get("seed")
    score = selection.get("selected_val_pearsonr")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GateError(f"validation run has no integer seed: {run_dir}")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise GateError(f"validation run has no finite selected score: {run_dir}")
    return seed, float(score)


def _paired_scores(run_dirs: Iterable[Path], *, label: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    for run_dir in run_dirs:
        seed, score = _validation_score(Path(run_dir))
        if seed in scores:
            raise GateError(f"duplicate {label} seed {seed}")
        scores[seed] = score
    return scores


def evaluate_validation_gate(
    baseline_runs: Sequence[Path],
    candidate_runs: Sequence[Path],
    *,
    minimum_wins: int = 2,
    minimum_mean_delta: float = 0.0,
    minimum_worst_seed_delta: float = -0.02,
) -> dict[str, Any]:
    """Evaluate a pre-test gate using only matched validation evidence."""

    if minimum_wins < 1:
        raise ValueError("minimum_wins must be positive")
    baseline = _paired_scores(baseline_runs, label="baseline")
    candidate = _paired_scores(candidate_runs, label="candidate")
    if not baseline or set(baseline) != set(candidate):
        raise GateError("baseline and candidate must contain the same non-empty seed set")

    rows = [
        {
            "seed": seed,
            "baseline_selected_val_pearsonr": baseline[seed],
            "candidate_selected_val_pearsonr": candidate[seed],
            "delta": candidate[seed] - baseline[seed],
        }
        for seed in sorted(baseline)
    ]
    deltas = [row["delta"] for row in rows]
    wins = sum(delta > 0.0 for delta in deltas)
    mean_delta = sum(deltas) / len(deltas)
    worst_delta = min(deltas)
    passed = (
        wins >= minimum_wins
        and mean_delta > minimum_mean_delta
        and worst_delta >= minimum_worst_seed_delta
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "per_seed": rows,
        "mean_delta": mean_delta,
        "wins": wins,
        "losses": len(deltas) - wins,
        "worst_seed_delta": worst_delta,
        "gate": {
            "minimum_wins": minimum_wins,
            "minimum_mean_delta": minimum_mean_delta,
            "minimum_worst_seed_delta": minimum_worst_seed_delta,
        },
        "test_status_before_final": "withheld",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", action="append", required=True, type=Path)
    parser.add_argument("--candidate-run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = evaluate_validation_gate(args.baseline_run, args.candidate_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
