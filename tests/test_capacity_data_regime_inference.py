from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.research.capacity_data_regime_inference import (
    CapacityExploratoryInferenceError,
    load_capacity_exploratory_inference,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/capacity_data_regime_exploratory_inference.json"


def test_exploratory_inference_config_is_strict_and_content_addressed() -> None:
    spec = load_capacity_exploratory_inference(CONFIG)

    assert spec.schema_version == 1
    assert spec.status == "exploratory"
    assert spec.alpha == pytest.approx(0.05)
    assert spec.tail == "greater"
    assert spec.zero_deltas == "include_as_nonnegative_exceedance"
    assert len(spec.cell_family_order) == 6
    assert len(spec.contrast_family_order) == 6
    assert spec.decision_rule.endswith("sealed_stable_improvement_decision")
    assert len(spec.sha256) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unknown": True}),
        lambda payload: payload.update({"alpha": 0.1}),
        lambda payload: payload.update({"tail": "two-sided"}),
        lambda payload: payload["cell_family_order"].pop(),
        lambda payload: payload["contrast_family_order"].reverse(),
    ],
)
def test_exploratory_inference_rejects_drift(tmp_path: Path, mutation) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "inference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CapacityExploratoryInferenceError):
        load_capacity_exploratory_inference(path)
