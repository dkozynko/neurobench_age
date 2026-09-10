from __future__ import annotations

import pytest

from neurobench_age.research.precision_simulation import (
    PrecisionScenario,
    PrecisionSimulationError,
    simulate_precision_scenario,
)


def test_precision_simulation_is_deterministic_and_reports_planning_only_summary() -> None:
    scenario = PrecisionScenario(
        name="small_effect",
        baseline_correlation=0.70,
        candidate_correlation=0.73,
        paired_error_correlation=0.80,
        seed_delta_sd=0.004,
        target_distribution="normal",
    )

    first = simulate_precision_scenario(
        scenario,
        subject_count=32,
        simulation_replicates=4,
        bootstrap_iterations=40,
        seed=20260910,
    )
    second = simulate_precision_scenario(
        scenario,
        subject_count=32,
        simulation_replicates=4,
        bootstrap_iterations=40,
        seed=20260910,
    )

    assert first == second
    assert first["status"] == "planning_only"
    assert first["subject_count"] == 32
    assert first["seed_count"] == 10
    assert first["simulation_replicates"] == 4
    assert first["bootstrap_iterations"] == 40
    assert 0.0 < first["valid_replicates"] <= 4
    assert first["ci_width_quantiles"]["q05"] >= 0.0
    assert first["ci_width_quantiles"]["q95"] >= first["ci_width_quantiles"]["q05"]
    assert first["assumptions"]["expected_contrast"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_correlation", 1.0),
        ("candidate_correlation", -1.0),
        ("paired_error_correlation", 1.1),
        ("seed_delta_sd", -0.1),
    ],
)
def test_precision_scenario_rejects_invalid_assumptions(field: str, value: float) -> None:
    kwargs = {
        "name": "invalid",
        "baseline_correlation": 0.7,
        "candidate_correlation": 0.73,
        "paired_error_correlation": 0.8,
        "seed_delta_sd": 0.0,
        field: value,
    }

    with pytest.raises(PrecisionSimulationError):
        PrecisionScenario(**kwargs)


def test_precision_simulation_rejects_too_small_cohort_or_replicates() -> None:
    scenario = PrecisionScenario(
        name="valid",
        baseline_correlation=0.70,
        candidate_correlation=0.73,
        paired_error_correlation=0.80,
        seed_delta_sd=0.0,
    )

    with pytest.raises(PrecisionSimulationError, match="subject_count"):
        simulate_precision_scenario(
            scenario,
            subject_count=1,
            simulation_replicates=4,
            bootstrap_iterations=40,
            seed=1,
        )
    with pytest.raises(PrecisionSimulationError, match="simulation_replicates"):
        simulate_precision_scenario(
            scenario,
            subject_count=32,
            simulation_replicates=0,
            bootstrap_iterations=40,
            seed=1,
        )
