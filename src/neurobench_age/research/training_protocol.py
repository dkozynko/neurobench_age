"""Strict NeuralBench-compatible contract for frozen-probe head training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPRESENTATION_PROTOCOL_SHA256 = (
    "66e0ae3139f2fb1da4ccef26fefd6c78678a47aa4ad2fcceb491eee11cb6cd21"
)


class FrozenProbeTrainingProtocolError(ValueError):
    """Raised when the final frozen-probe training contract drifts."""


@dataclass(frozen=True)
class BatchingContract:
    unit: str
    shuffle: str
    drop_last: bool


@dataclass(frozen=True)
class OptimizerContract:
    name: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerContract:
    name: str
    max_learning_rate: float
    pct_start: float
    anneal_strategy: str
    div_factor: float
    final_div_factor: float
    interval: str
    frequency: int


@dataclass(frozen=True)
class FrozenProbeTrainingProtocol:
    schema_version: int
    study_id: str
    status: str
    representation_protocol_sha256: str
    seeds: tuple[int, ...]
    batching: BatchingContract
    optimizer: OptimizerContract
    scheduler: SchedulerContract
    gradient_clip_norm: float
    batch_size: int
    max_epochs: int
    patience: int
    loss: str
    checkpoint_metric: str
    metric_mode: str
    sha256: str


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrozenProbeTrainingProtocolError(f"{path} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise FrozenProbeTrainingProtocolError(
            f"{path} has unknown fields={unknown} missing fields={missing}"
        )


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise FrozenProbeTrainingProtocolError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrozenProbeTrainingProtocolError(f"{path} must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenProbeTrainingProtocolError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FrozenProbeTrainingProtocolError(f"{path} must be finite")
    return result


def load_frozen_probe_training_protocol(
    path: Path,
    *,
    profile: str = "primary",
) -> FrozenProbeTrainingProtocol:
    """Load and fail closed against the approved final training contract."""

    if profile not in {"primary", "layerwise"}:
        raise FrozenProbeTrainingProtocolError("unknown frozen-probe training profile")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenProbeTrainingProtocolError(
            f"could not read frozen-probe training protocol: {path}"
        ) from error
    root = _object(payload, "training protocol")
    root_fields = {
        "schema_version",
        "study_id",
        "status",
        "representation_protocol_sha256",
        "seeds",
        "batching",
        "optimizer",
        "scheduler",
        "gradient_clip_norm",
        "batch_size",
        "max_epochs",
        "patience",
        "loss",
        "checkpoint_metric",
        "metric_mode",
    }
    _exact_fields(root, root_fields, "training protocol")

    batching = _object(root["batching"], "batching")
    _exact_fields(batching, {"unit", "shuffle", "drop_last"}, "batching")
    optimizer = _object(root["optimizer"], "optimizer")
    _exact_fields(
        optimizer, {"name", "learning_rate", "weight_decay"}, "optimizer"
    )
    scheduler = _object(root["scheduler"], "scheduler")
    _exact_fields(
        scheduler,
        {
            "name",
            "max_learning_rate",
            "pct_start",
            "anneal_strategy",
            "div_factor",
            "final_div_factor",
            "interval",
            "frequency",
        },
        "scheduler",
    )
    raw_seeds = root["seeds"]
    if not isinstance(raw_seeds, list):
        raise FrozenProbeTrainingProtocolError("seeds must be an array")

    result = FrozenProbeTrainingProtocol(
        schema_version=_integer(root["schema_version"], "schema_version"),
        study_id=_string(root["study_id"], "study_id"),
        status=_string(root["status"], "status"),
        representation_protocol_sha256=_string(
            root["representation_protocol_sha256"],
            "representation_protocol_sha256",
        ),
        seeds=tuple(_integer(seed, "seeds[]") for seed in raw_seeds),
        batching=BatchingContract(
            unit=_string(batching["unit"], "batching.unit"),
            shuffle=_string(batching["shuffle"], "batching.shuffle"),
            drop_last=batching["drop_last"],
        ),
        optimizer=OptimizerContract(
            name=_string(optimizer["name"], "optimizer.name"),
            learning_rate=_number(
                optimizer["learning_rate"], "optimizer.learning_rate"
            ),
            weight_decay=_number(optimizer["weight_decay"], "optimizer.weight_decay"),
        ),
        scheduler=SchedulerContract(
            name=_string(scheduler["name"], "scheduler.name"),
            max_learning_rate=_number(
                scheduler["max_learning_rate"], "scheduler.max_learning_rate"
            ),
            pct_start=_number(scheduler["pct_start"], "scheduler.pct_start"),
            anneal_strategy=_string(
                scheduler["anneal_strategy"], "scheduler.anneal_strategy"
            ),
            div_factor=_number(scheduler["div_factor"], "scheduler.div_factor"),
            final_div_factor=_number(
                scheduler["final_div_factor"], "scheduler.final_div_factor"
            ),
            interval=_string(scheduler["interval"], "scheduler.interval"),
            frequency=_integer(scheduler["frequency"], "scheduler.frequency"),
        ),
        gradient_clip_norm=_number(
            root["gradient_clip_norm"], "gradient_clip_norm"
        ),
        batch_size=_integer(root["batch_size"], "batch_size"),
        max_epochs=_integer(root["max_epochs"], "max_epochs"),
        patience=_integer(root["patience"], "patience"),
        loss=_string(root["loss"], "loss"),
        checkpoint_metric=_string(
            root["checkpoint_metric"], "checkpoint_metric"
        ),
        metric_mode=_string(root["metric_mode"], "metric_mode"),
        sha256=_canonical_sha256(root),
    )
    if not isinstance(result.batching.drop_last, bool):
        raise FrozenProbeTrainingProtocolError("batching.drop_last must be boolean")
    if result.schema_version != 1 or result.status != "final":
        raise FrozenProbeTrainingProtocolError(
            "training protocol must be schema 1 with final status"
        )
    if profile == "primary" and result.representation_protocol_sha256 != REPRESENTATION_PROTOCOL_SHA256:
        raise FrozenProbeTrainingProtocolError(
            "training protocol representation protocol reference is not approved"
        )
    if result.seeds != tuple(range(33, 43)):
        raise FrozenProbeTrainingProtocolError(
            "training protocol must use exactly seeds 33 through 42"
        )
    if (
        result.batching.unit != "global_window"
        or result.batching.shuffle != "seeded_randperm"
    ):
        raise FrozenProbeTrainingProtocolError(
            "training batching must use seeded global window permutation"
        )
    if result.batching.drop_last:
        raise FrozenProbeTrainingProtocolError("training batching drop_last must be false")
    if result.optimizer.name != "AdamW":
        raise FrozenProbeTrainingProtocolError("optimizer must be AdamW")
    if result.optimizer.learning_rate != 1e-4:
        raise FrozenProbeTrainingProtocolError(
            "optimizer learning rate must be 1e-4"
        )
    if result.optimizer.weight_decay != 0.05:
        raise FrozenProbeTrainingProtocolError("optimizer weight decay must be 0.05")
    if result.scheduler.name != "OneCycleLR":
        raise FrozenProbeTrainingProtocolError("scheduler must be OneCycleLR")
    if result.scheduler.max_learning_rate != 1e-4:
        raise FrozenProbeTrainingProtocolError(
            "scheduler maximum learning rate must be 1e-4"
        )
    if result.scheduler.pct_start != 0.1:
        raise FrozenProbeTrainingProtocolError("scheduler pct_start must be 0.1")
    if result.scheduler.anneal_strategy != "cos":
        raise FrozenProbeTrainingProtocolError(
            "scheduler anneal strategy must be cosine"
        )
    if result.scheduler.div_factor != 25.0:
        raise FrozenProbeTrainingProtocolError("scheduler div_factor must be 25")
    if result.scheduler.final_div_factor != 10_000.0:
        raise FrozenProbeTrainingProtocolError(
            "scheduler final_div_factor must be 10000"
        )
    if result.scheduler.interval != "step" or result.scheduler.frequency != 1:
        raise FrozenProbeTrainingProtocolError(
            "OneCycleLR must step once per optimizer step"
        )
    if result.gradient_clip_norm != 1.0:
        raise FrozenProbeTrainingProtocolError(
            "gradient clipping norm must be exactly 1.0"
        )
    if result.batch_size != 64:
        raise FrozenProbeTrainingProtocolError("batch_size must be exactly 64")
    if result.max_epochs != 40:
        raise FrozenProbeTrainingProtocolError("training budget must be exactly 40 epochs")
    if result.patience != 7:
        raise FrozenProbeTrainingProtocolError("early stopping must use patience 7")
    if result.loss != "MSELoss":
        raise FrozenProbeTrainingProtocolError("loss must be MSELoss")
    if (
        result.checkpoint_metric != "validation_pearson"
        or result.metric_mode != "max"
    ):
        raise FrozenProbeTrainingProtocolError(
            "checkpoint selection must maximize validation Pearson"
        )
    return result
