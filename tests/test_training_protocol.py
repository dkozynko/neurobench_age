from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobench_age.research.training_protocol import (
    FrozenProbeTrainingProtocolError,
    load_frozen_probe_training_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATION_PROTOCOL_PATH = (
    ROOT / "configs" / "research" / "external_frozen_probe.json"
)
TRAINING_PROTOCOL_PATH = (
    ROOT / "configs" / "research" / "neuralbench_frozen_probe_training.json"
)
REPRESENTATION_PROTOCOL_SHA256 = (
    "66e0ae3139f2fb1da4ccef26fefd6c78678a47aa4ad2fcceb491eee11cb6cd21"
)


def _payload() -> dict[str, object]:
    return json.loads(TRAINING_PROTOCOL_PATH.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "training_protocol.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_final_training_protocol_matches_neuralbench_contract() -> None:
    protocol = load_frozen_probe_training_protocol(TRAINING_PROTOCOL_PATH)

    assert protocol.schema_version == 1
    assert protocol.status == "final"
    assert protocol.representation_protocol_sha256 == REPRESENTATION_PROTOCOL_SHA256
    assert protocol.seeds == tuple(range(33, 43))
    assert protocol.batching.unit == "global_window"
    assert protocol.batching.shuffle == "seeded_randperm"
    assert protocol.batching.drop_last is False
    assert protocol.optimizer.name == "AdamW"
    assert protocol.optimizer.learning_rate == 1e-4
    assert protocol.optimizer.weight_decay == 0.05
    assert protocol.scheduler.name == "OneCycleLR"
    assert protocol.scheduler.max_learning_rate == 1e-4
    assert protocol.scheduler.pct_start == 0.1
    assert protocol.scheduler.anneal_strategy == "cos"
    assert protocol.scheduler.div_factor == 25.0
    assert protocol.scheduler.final_div_factor == 10_000.0
    assert protocol.scheduler.interval == "step"
    assert protocol.scheduler.frequency == 1
    assert protocol.gradient_clip_norm == 1.0
    assert protocol.batch_size == 64
    assert protocol.max_epochs == 40
    assert protocol.patience == 7
    assert protocol.loss == "MSELoss"
    assert protocol.checkpoint_metric == "validation_pearson"
    assert protocol.metric_mode == "max"
    assert len(protocol.sha256) == 64


def test_training_protocol_digest_is_canonical_across_json_formatting(
    tmp_path: Path,
) -> None:
    protocol = load_frozen_probe_training_protocol(TRAINING_PROTOCOL_PATH)
    reformatted = tmp_path / "training_protocol.json"
    reformatted.write_text(json.dumps(_payload(), indent=7), encoding="utf-8")

    assert load_frozen_probe_training_protocol(reformatted).sha256 == protocol.sha256


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"unexpected": True}), "unknown fields"),
        (
            lambda payload: payload.update(
                {"representation_protocol_sha256": "0" * 64}
            ),
            "representation protocol",
        ),
        (lambda payload: payload.update({"seeds": [33, 34]}), "seeds 33 through 42"),
        (
            lambda payload: payload["batching"].update({"unit": "subject"}),
            "global window",
        ),
        (
            lambda payload: payload["batching"].update({"drop_last": True}),
            "drop_last",
        ),
        (
            lambda payload: payload["optimizer"].update({"learning_rate": 1e-3}),
            "learning rate",
        ),
        (
            lambda payload: payload["optimizer"].update({"weight_decay": 1e-4}),
            "weight decay",
        ),
        (
            lambda payload: payload["scheduler"].update({"name": "none"}),
            "OneCycleLR",
        ),
        (
            lambda payload: payload["scheduler"].update({"pct_start": 0.2}),
            "pct_start",
        ),
        (
            lambda payload: payload.update({"gradient_clip_norm": 0.0}),
            "gradient clipping",
        ),
        (lambda payload: payload.update({"max_epochs": 100}), "40 epochs"),
        (lambda payload: payload.update({"patience": 15}), "patience 7"),
        (
            lambda payload: payload["scheduler"].update(
                {"max_learning_rate": float("nan")}
            ),
            "finite",
        ),
    ],
)
def test_training_protocol_rejects_drift(
    tmp_path: Path, mutate: object, message: str
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(FrozenProbeTrainingProtocolError, match=message):
        load_frozen_probe_training_protocol(_write(tmp_path, payload))


def test_training_protocol_rejects_boolean_numeric_values(tmp_path: Path) -> None:
    payload = _payload()
    payload["batch_size"] = True

    with pytest.raises(FrozenProbeTrainingProtocolError, match="batch_size"):
        load_frozen_probe_training_protocol(_write(tmp_path, payload))
