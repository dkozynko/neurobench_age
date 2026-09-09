from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.research.capacity_data_regime import (
    CapacityDataRegimeProtocolError,
    load_capacity_data_regime_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "research" / "capacity_data_regime.json"
TRAINING_PROTOCOL_SHA256 = (
    "721f745080e262d7813d2a67056643afae89dbf1b259524e6d5d3951bba3b4d6"
)
REPRESENTATION_PROTOCOL_FILE_SHA256 = (
    "654923b15f9d4f4ccec0eb02637898b289bd8879326c71a96e5f6d88e698c60f"
)


def _payload() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "capacity_data_regime.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_capacity_data_regime_protocol_is_exactly_predeclared() -> None:
    protocol = load_capacity_data_regime_protocol(PROTOCOL_PATH, repository_root=ROOT)

    assert protocol.schema_version == 1
    assert protocol.status == "final"
    assert protocol.extension_id == "reve_age_capacity_data_regime_v1"
    assert protocol.training_sizes == (200, 400, 800)
    assert protocol.future_endpoints == ("all_available",)
    assert protocol.seeds == tuple(range(33, 43))
    assert protocol.eligible_training_releases == (
        "R1",
        "R2",
        "R3",
        "R4",
        "R6",
        "R7",
        "R9",
        "R10",
    )
    assert protocol.validation_release == "R8"
    assert protocol.test_release == "R5"
    assert protocol.priority_salt == "capacity-data-regime-v1"
    assert protocol.head_names == (
        "mean_linear",
        "mean_rich_stats_residual",
        "mean_mlp_residual_matched(hidden_dim=4)",
    )
    assert protocol.training_protocol_path == Path(
        "configs/research/neuralbench_frozen_probe_training.json"
    )
    assert protocol.training_protocol_file_sha256 == TRAINING_PROTOCOL_SHA256
    assert protocol.representation_protocol_path == Path(
        "configs/research/external_frozen_probe.json"
    )
    assert (
        protocol.representation_protocol_file_sha256
        == REPRESENTATION_PROTOCOL_FILE_SHA256
    )
    assert protocol.bootstrap_iterations == 10_000
    assert protocol.bootstrap_seed == 20260909
    assert protocol.minimum_valid_bootstrap_replicates == 9_950


def test_capacity_data_regime_protocol_exposes_only_training_contract_values() -> None:
    protocol = load_capacity_data_regime_protocol(PROTOCOL_PATH, repository_root=ROOT)

    assert protocol.training_protocol_source == "configs/research/neuralbench_frozen_probe_training.json"
    assert protocol.representation_protocol_source == "configs/research/external_frozen_probe.json"
    assert protocol.external_primary_subject_count == 75
    assert protocol.expected_run_count == 90
    assert protocol.expected_prediction_count == 6_750
    assert protocol.allow_extrapolation is False
    assert protocol.free_space_floor_bytes == 12 * 1024**3
    assert protocol.free_space_cache_fraction == 0.25
    assert protocol.free_space_output_multiplier == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload.update({"training_sizes": [200, 800]}),
        lambda payload: payload.update({"seeds": [33, 34]}),
        lambda payload: payload.update({"validation_release": "R5"}),
        lambda payload: payload.update({"eligible_training_releases": ["R8"]}),
        lambda payload: payload["bootstrap"].update({"seed": 20260903}),
        lambda payload: payload["training_protocol"].update({"sha256": "0" * 64}),
    ],
)
def test_capacity_data_regime_protocol_rejects_drift(
    tmp_path: Path, mutation: object
) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(CapacityDataRegimeProtocolError):
        load_capacity_data_regime_protocol(_write(tmp_path, payload), repository_root=ROOT)


def test_capacity_data_regime_protocol_digest_is_format_independent(
    tmp_path: Path,
) -> None:
    protocol = load_capacity_data_regime_protocol(PROTOCOL_PATH, repository_root=ROOT)
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(_payload(), indent=7), encoding="utf-8")

    assert (
        load_capacity_data_regime_protocol(reformatted, repository_root=ROOT).sha256
        == protocol.sha256
    )
