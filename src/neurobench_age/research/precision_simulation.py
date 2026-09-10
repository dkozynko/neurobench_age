"""Planning-only precision simulation for the external head contrast.

The simulator mirrors the study's hierarchical paired bootstrap at the level
of synthetic subject predictions. It is deliberately separate from inference:
it never reads participant metadata, EEG, checkpoints, or observed predictions.
Its output is therefore sensitivity evidence for choosing a precision rule, not
an approval of the external study.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


EXPECTED_SEEDS = tuple(range(33, 43))


class PrecisionSimulationError(ValueError):
    """Raised when planning-simulation assumptions are invalid."""


@dataclass(frozen=True)
class PrecisionScenario:
    """Synthetic assumptions for one external precision scenario."""

    name: str
    baseline_correlation: float
    candidate_correlation: float
    paired_error_correlation: float
    seed_delta_sd: float
    target_distribution: str = "normal"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PrecisionSimulationError("scenario name must be non-empty")
        for field in (
            "baseline_correlation",
            "candidate_correlation",
            "paired_error_correlation",
            "seed_delta_sd",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PrecisionSimulationError(f"{field} must be numeric")
            if not math.isfinite(float(value)):
                raise PrecisionSimulationError(f"{field} must be finite")
        if not -1.0 < float(self.baseline_correlation) < 1.0:
            raise PrecisionSimulationError("baseline_correlation must be strictly between -1 and 1")
        if not -1.0 < float(self.candidate_correlation) < 1.0:
            raise PrecisionSimulationError("candidate_correlation must be strictly between -1 and 1")
        if not -1.0 < float(self.paired_error_correlation) < 1.0:
            raise PrecisionSimulationError(
                "paired_error_correlation must be strictly between -1 and 1"
            )
        if float(self.seed_delta_sd) < 0.0:
            raise PrecisionSimulationError("seed_delta_sd must be non-negative")
        if abs(float(self.candidate_correlation)) + 4.0 * float(self.seed_delta_sd) >= 1.0:
            raise PrecisionSimulationError(
                "candidate_correlation plus four seed_delta_sd must remain inside (-1, 1)"
            )
        if self.target_distribution not in {"normal", "uniform"}:
            raise PrecisionSimulationError(
                "target_distribution must be 'normal' or 'uniform'"
            )


def _standardize(values: np.ndarray, *, name: str) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    scale = float(np.sqrt(np.mean(np.square(centered))))
    if scale <= 1e-12 or not math.isfinite(scale):
        raise PrecisionSimulationError(f"{name} cannot be standardized")
    return centered / scale


def _pearson_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute Pearson correlations for arrays whose final axis is subjects."""

    left_centered = left - np.mean(left, axis=-1, keepdims=True)
    right_centered = right - np.mean(right, axis=-1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=-1)
    denominator = np.sqrt(
        np.sum(np.square(left_centered), axis=-1)
        * np.sum(np.square(right_centered), axis=-1)
    )
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _draw_target(rng: np.random.Generator, *, count: int, distribution: str) -> np.ndarray:
    if distribution == "normal":
        raw = rng.normal(size=count)
    elif distribution == "uniform":
        raw = rng.uniform(-1.0, 1.0, size=count)
    else:  # pragma: no cover - guarded by PrecisionScenario
        raise PrecisionSimulationError(f"unsupported target distribution: {distribution}")
    return _standardize(raw, name="synthetic target")


def _bootstrap_contrast(
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Return percentile bounds for the same paired seed/subject bootstrap."""

    seed_count, subject_count = baseline.shape
    sampled_seed = rng.integers(0, seed_count, size=(iterations, seed_count))
    sampled_subject = rng.integers(0, subject_count, size=(iterations, subject_count))
    target_sample = target[sampled_subject]
    baseline_sample = baseline[
        sampled_seed[:, :, None], sampled_subject[:, None, :]
    ]
    candidate_sample = candidate[
        sampled_seed[:, :, None], sampled_subject[:, None, :]
    ]
    target_for_seed = target_sample[:, None, :]
    baseline_correlations = _pearson_batch(target_for_seed, baseline_sample)
    candidate_correlations = _pearson_batch(target_for_seed, candidate_sample)
    contrasts = np.mean(candidate_correlations - baseline_correlations, axis=1)
    finite = contrasts[np.isfinite(contrasts)]
    if finite.size == 0:
        raise PrecisionSimulationError("all bootstrap iterations were degenerate")
    alpha = 0.05
    return (
        float(np.quantile(finite, alpha / 2.0)),
        float(np.quantile(finite, 1.0 - alpha / 2.0)),
        int(iterations - finite.size),
    )


def _simulate_one_replicate(
    scenario: PrecisionScenario,
    *,
    subject_count: int,
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    target = _draw_target(
        rng,
        count=subject_count,
        distribution=scenario.target_distribution,
    )
    baseline = np.empty((len(EXPECTED_SEEDS), subject_count), dtype=np.float64)
    candidate = np.empty_like(baseline)
    shared_noise_weight = float(scenario.paired_error_correlation)
    independent_noise_weight = math.sqrt(1.0 - shared_noise_weight**2)
    for seed_index in range(len(EXPECTED_SEEDS)):
        baseline_noise = rng.normal(size=subject_count)
        candidate_noise = (
            shared_noise_weight * baseline_noise
            + independent_noise_weight * rng.normal(size=subject_count)
        )
        candidate_correlation = float(
            scenario.candidate_correlation
            + rng.normal(scale=scenario.seed_delta_sd)
        )
        if not -1.0 < candidate_correlation < 1.0:
            raise PrecisionSimulationError(
                "sampled candidate correlation left the valid interval; "
                "reduce seed_delta_sd"
            )
        baseline[seed_index] = (
            scenario.baseline_correlation * target
            + math.sqrt(1.0 - scenario.baseline_correlation**2) * baseline_noise
        )
        candidate[seed_index] = (
            candidate_correlation * target
            + math.sqrt(1.0 - candidate_correlation**2) * candidate_noise
        )

    baseline_correlations = _pearson_batch(
        np.broadcast_to(target, baseline.shape), baseline
    )
    candidate_correlations = _pearson_batch(
        np.broadcast_to(target, candidate.shape), candidate
    )
    point_estimate = float(np.mean(candidate_correlations - baseline_correlations))
    ci_low, ci_high, failed_bootstrap = _bootstrap_contrast(
        target,
        baseline,
        candidate,
        iterations=bootstrap_iterations,
        rng=rng,
    )
    return point_estimate, ci_high - ci_low, float(ci_low <= scenario.candidate_correlation - scenario.baseline_correlation <= ci_high), failed_bootstrap


def simulate_precision_scenario(
    scenario: PrecisionScenario,
    *,
    subject_count: int,
    simulation_replicates: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate CI-width sensitivity under one synthetic scenario."""

    if isinstance(subject_count, bool) or not isinstance(subject_count, int) or subject_count < 3:
        raise PrecisionSimulationError("subject_count must be an integer >= 3")
    if (
        isinstance(simulation_replicates, bool)
        or not isinstance(simulation_replicates, int)
        or simulation_replicates <= 0
    ):
        raise PrecisionSimulationError("simulation_replicates must be a positive integer")
    if (
        isinstance(bootstrap_iterations, bool)
        or not isinstance(bootstrap_iterations, int)
        or bootstrap_iterations < 10
    ):
        raise PrecisionSimulationError("bootstrap_iterations must be an integer >= 10")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PrecisionSimulationError("seed must be an integer")

    rng = np.random.default_rng(seed)
    point_estimates: list[float] = []
    ci_widths: list[float] = []
    coverage: list[float] = []
    failed_bootstrap = 0
    for _ in range(simulation_replicates):
        point, width, covered, failed = _simulate_one_replicate(
            scenario,
            subject_count=subject_count,
            bootstrap_iterations=bootstrap_iterations,
            rng=rng,
        )
        point_estimates.append(point)
        ci_widths.append(width)
        coverage.append(covered)
        failed_bootstrap += failed

    width_array = np.asarray(ci_widths, dtype=np.float64)
    point_array = np.asarray(point_estimates, dtype=np.float64)
    return {
        "status": "planning_only",
        "scenario": scenario.name,
        "subject_count": subject_count,
        "seed_ids": list(EXPECTED_SEEDS),
        "seed_count": len(EXPECTED_SEEDS),
        "simulation_replicates": simulation_replicates,
        "bootstrap_iterations": bootstrap_iterations,
        "assumptions": {
            "baseline_correlation": float(scenario.baseline_correlation),
            "candidate_correlation": float(scenario.candidate_correlation),
            "expected_contrast": float(
                scenario.candidate_correlation - scenario.baseline_correlation
            ),
            "paired_error_correlation": float(scenario.paired_error_correlation),
            "seed_delta_sd": float(scenario.seed_delta_sd),
            "target_distribution": scenario.target_distribution,
        },
        "point_estimate": {
            "mean": float(point_array.mean()),
            "q05": float(np.quantile(point_array, 0.05)),
            "q95": float(np.quantile(point_array, 0.95)),
        },
        "ci_width_quantiles": {
            "q05": float(np.quantile(width_array, 0.05)),
            "median": float(np.quantile(width_array, 0.50)),
            "q95": float(np.quantile(width_array, 0.95)),
        },
        "expected_contrast_coverage": float(np.mean(coverage)),
        "valid_replicates": int(len(ci_widths)),
        "failed_bootstrap_iterations": int(failed_bootstrap),
    }


def scenario_from_mapping(value: Any) -> PrecisionScenario:
    if not isinstance(value, dict):
        raise PrecisionSimulationError("each precision scenario must be an object")
    allowed = set(asdict(PrecisionScenario("x", 0.0, 0.1, 0.0, 0.0)))
    unknown = set(value) - allowed
    if unknown:
        raise PrecisionSimulationError(
            f"precision scenario has unknown fields: {sorted(unknown)}"
        )
    required = allowed - {"target_distribution"}
    missing = required - set(value)
    if missing:
        raise PrecisionSimulationError(
            f"precision scenario is missing fields: {sorted(missing)}"
        )
    return PrecisionScenario(**value)


def simulate_precision_scenarios(
    scenarios: Sequence[PrecisionScenario],
    *,
    subject_count: int,
    simulation_replicates: int,
    bootstrap_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run independent deterministic substreams for an ordered scenario list."""

    if not scenarios:
        raise PrecisionSimulationError("at least one precision scenario is required")
    master = np.random.SeedSequence(seed)
    children = master.spawn(len(scenarios))
    return [
        simulate_precision_scenario(
            scenario,
            subject_count=subject_count,
            simulation_replicates=simulation_replicates,
            bootstrap_iterations=bootstrap_iterations,
            seed=int(child.generate_state(1, dtype=np.uint64)[0]),
        )
        for scenario, child in zip(scenarios, children, strict=True)
    ]
