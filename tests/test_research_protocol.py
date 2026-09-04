from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.research.protocol import ProtocolError, load_study_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "research" / "external_frozen_probe.json"


def _payload() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_approved_protocol_loads_with_predeclared_primary_contract() -> None:
    protocol = load_study_protocol(PROTOCOL_PATH)

    assert protocol.seeds == tuple(range(33, 43))
    assert protocol.head_names == (
        "mean_linear",
        "mean_layer_linear",
        "mean_rich_stats_residual",
        "multi_query_rich_stats",
    )
    assert protocol.encoder.frozen is True
    assert protocol.encoder.layer_indices == (-2, -1)
    assert protocol.statistics.bootstrap_iterations == 10_000
    assert protocol.statistics.bootstrap_seed == 20260903
    assert protocol.statistics.minimum_seed_wins == 8
    assert protocol.statistics.minimum_worst_seed_delta == -0.01
    assert len(protocol.sha256) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["encoder"].update({"unexpected": True}),
    ],
)
def test_protocol_rejects_unknown_fields(tmp_path: Path, mutation: object) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(ProtocolError, match="unknown fields"):
        load_study_protocol(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["training"].update({"seeds": [33, 34]}), "seeds 33 through 42"),
        (lambda payload: payload["heads"].append({"name": "new_head", "layer_index": -1, "aggregation": "mean"}), "exactly four"),
        (lambda payload: payload["encoder"].update({"frozen": False}), "frozen"),
        (lambda payload: payload["encoder"].update({"layer_indices": [-3, -1]}), "layers -2 and -1"),
        (lambda payload: payload["statistics"].update({"bootstrap_iterations": 999}), "10,000"),
    ],
)
def test_protocol_rejects_changes_to_confirmatory_invariants(
    tmp_path: Path, mutate: object, message: str
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(ProtocolError, match=message):
        load_study_protocol(_write(tmp_path, payload))


def test_protocol_digest_is_canonical_across_json_formatting(tmp_path: Path) -> None:
    protocol = load_study_protocol(PROTOCOL_PATH)
    reformatted = tmp_path / "protocol.json"
    reformatted.write_text(json.dumps(_payload(), indent=7), encoding="utf-8")

    assert load_study_protocol(reformatted).sha256 == protocol.sha256
