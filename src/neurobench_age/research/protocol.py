"""Strict executable protocol for the external frozen-REVE study."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class ProtocolError(ValueError):
    """Raised when a study protocol is incomplete or has drifted."""


def _object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{path} must be an object")
    return value


def _expect_keys(
    value: Mapping[str, Any], *, path: str, required: Sequence[str]
) -> None:
    required_set = set(required)
    unknown = sorted(set(value) - required_set)
    missing = sorted(required_set - set(value))
    if unknown:
        raise ProtocolError(f"{path} has unknown fields: {unknown}")
    if missing:
        raise ProtocolError(f"{path} is missing fields: {missing}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{path} must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{path} must be numeric")
    return float(value)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{path} must be boolean")
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ProtocolError(f"{path} must be an array")
    return value


@dataclass(frozen=True)
class DatasetContract:
    development_name: str
    development_use: str
    development_test_access: str
    external_name: str
    external_nemar_id: str
    external_release: str
    pilot_size: int
    pilot_salt: str
    primary_cohort: str
    extrapolation_cohort: str
    minimum_primary_subjects: int


@dataclass(frozen=True)
class EncoderContract:
    checkpoint: str
    frozen: bool
    eval_mode: bool
    inference_mode: bool
    layer_indices: tuple[int, ...]


@dataclass(frozen=True)
class PreprocessingContract:
    sample_rate_hz: float
    bandpass_hz: tuple[float, float]
    notch_hz: tuple[float, ...] | None
    scaler: str
    clamp: float
    window_seconds: float
    stride_seconds: float
    max_seconds_per_subject: float
    cross_block_windows: bool
    spatial_interpolation: bool
    subject_aggregation: str


@dataclass(frozen=True)
class HeadContract:
    name: str
    layer_index: int
    aggregation: str


@dataclass(frozen=True)
class TrainingContract:
    seeds: tuple[int, ...]
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    loss: str
    checkpoint_metric: str
    metric_mode: str


@dataclass(frozen=True)
class StatisticsContract:
    bootstrap_iterations: int
    bootstrap_seed: int
    confidence: float
    randomization_tail: str
    holm_order: tuple[str, ...]
    alpha: float
    minimum_seed_wins: int
    minimum_worst_seed_delta: float
    require_ci_above_zero: bool


@dataclass(frozen=True)
class StudyProtocol:
    schema_version: int
    study_id: str
    status: str
    datasets: DatasetContract
    encoder: EncoderContract
    preprocessing: PreprocessingContract
    heads: tuple[HeadContract, ...]
    training: TrainingContract
    statistics: StatisticsContract
    sha256: str

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.training.seeds

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(head.name for head in self.heads)


def _parse_datasets(value: object) -> DatasetContract:
    datasets = _object(value, "datasets")
    _expect_keys(datasets, path="datasets", required=("development", "external"))
    development = _object(datasets["development"], "datasets.development")
    _expect_keys(
        development,
        path="datasets.development",
        required=("name", "use", "test_access"),
    )
    external = _object(datasets["external"], "datasets.external")
    _expect_keys(
        external,
        path="datasets.external",
        required=(
            "name",
            "nemar_id",
            "release",
            "pilot_size",
            "pilot_salt",
            "primary_cohort",
            "extrapolation_cohort",
            "minimum_primary_subjects",
        ),
    )
    result = DatasetContract(
        development_name=_string(development["name"], "datasets.development.name"),
        development_use=_string(development["use"], "datasets.development.use"),
        development_test_access=_string(
            development["test_access"], "datasets.development.test_access"
        ),
        external_name=_string(external["name"], "datasets.external.name"),
        external_nemar_id=_string(external["nemar_id"], "datasets.external.nemar_id"),
        external_release=_string(external["release"], "datasets.external.release"),
        pilot_size=_integer(external["pilot_size"], "datasets.external.pilot_size"),
        pilot_salt=_string(external["pilot_salt"], "datasets.external.pilot_salt"),
        primary_cohort=_string(external["primary_cohort"], "datasets.external.primary_cohort"),
        extrapolation_cohort=_string(
            external["extrapolation_cohort"], "datasets.external.extrapolation_cohort"
        ),
        minimum_primary_subjects=_integer(
            external["minimum_primary_subjects"],
            "datasets.external.minimum_primary_subjects",
        ),
    )
    if result.external_nemar_id != "nm000153" or result.pilot_size != 10:
        raise ProtocolError("external dataset must be MIPDB nm000153 with a 10-subject pilot")
    if result.minimum_primary_subjects != 50:
        raise ProtocolError("minimum primary cohort must be 50 subjects")
    return result


def _parse_encoder(value: object) -> EncoderContract:
    encoder = _object(value, "encoder")
    _expect_keys(
        encoder,
        path="encoder",
        required=("checkpoint", "frozen", "eval_mode", "inference_mode", "layer_indices"),
    )
    result = EncoderContract(
        checkpoint=_string(encoder["checkpoint"], "encoder.checkpoint"),
        frozen=_boolean(encoder["frozen"], "encoder.frozen"),
        eval_mode=_boolean(encoder["eval_mode"], "encoder.eval_mode"),
        inference_mode=_boolean(encoder["inference_mode"], "encoder.inference_mode"),
        layer_indices=tuple(
            _integer(item, "encoder.layer_indices[]")
            for item in _sequence(encoder["layer_indices"], "encoder.layer_indices")
        ),
    )
    if result.checkpoint != "brain-bzh/reve-base":
        raise ProtocolError("encoder checkpoint must be brain-bzh/reve-base")
    if not (result.frozen and result.eval_mode and result.inference_mode):
        raise ProtocolError("encoder must be frozen and run in eval/inference mode")
    if result.layer_indices != (-2, -1):
        raise ProtocolError("encoder cache must contain exactly layers -2 and -1")
    return result


def _parse_preprocessing(value: object) -> PreprocessingContract:
    preprocessing = _object(value, "preprocessing")
    keys = (
        "sample_rate_hz",
        "bandpass_hz",
        "notch_hz",
        "scaler",
        "clamp",
        "window_seconds",
        "stride_seconds",
        "max_seconds_per_subject",
        "cross_block_windows",
        "spatial_interpolation",
        "subject_aggregation",
    )
    _expect_keys(preprocessing, path="preprocessing", required=keys)
    bandpass = tuple(
        _number(item, "preprocessing.bandpass_hz[]")
        for item in _sequence(preprocessing["bandpass_hz"], "preprocessing.bandpass_hz")
    )
    if len(bandpass) != 2:
        raise ProtocolError("preprocessing.bandpass_hz must contain two values")
    notch_raw = preprocessing["notch_hz"]
    notch = None if notch_raw is None else tuple(
        _number(item, "preprocessing.notch_hz[]")
        for item in _sequence(notch_raw, "preprocessing.notch_hz")
    )
    result = PreprocessingContract(
        sample_rate_hz=_number(preprocessing["sample_rate_hz"], "preprocessing.sample_rate_hz"),
        bandpass_hz=(bandpass[0], bandpass[1]),
        notch_hz=notch,
        scaler=_string(preprocessing["scaler"], "preprocessing.scaler"),
        clamp=_number(preprocessing["clamp"], "preprocessing.clamp"),
        window_seconds=_number(preprocessing["window_seconds"], "preprocessing.window_seconds"),
        stride_seconds=_number(preprocessing["stride_seconds"], "preprocessing.stride_seconds"),
        max_seconds_per_subject=_number(
            preprocessing["max_seconds_per_subject"],
            "preprocessing.max_seconds_per_subject",
        ),
        cross_block_windows=_boolean(
            preprocessing["cross_block_windows"], "preprocessing.cross_block_windows"
        ),
        spatial_interpolation=_boolean(
            preprocessing["spatial_interpolation"], "preprocessing.spatial_interpolation"
        ),
        subject_aggregation=_string(
            preprocessing["subject_aggregation"], "preprocessing.subject_aggregation"
        ),
    )
    expected = (200.0, (0.5, 99.5), None, "StandardScaler", 15.0, 2.0, 2.0, 120.0, False, False, "arithmetic_mean")
    actual = (
        result.sample_rate_hz,
        result.bandpass_hz,
        result.notch_hz,
        result.scaler,
        result.clamp,
        result.window_seconds,
        result.stride_seconds,
        result.max_seconds_per_subject,
        result.cross_block_windows,
        result.spatial_interpolation,
        result.subject_aggregation,
    )
    if actual != expected:
        raise ProtocolError("preprocessing does not match the approved common input contract")
    return result


def _parse_heads(value: object) -> tuple[HeadContract, ...]:
    raw_heads = _sequence(value, "heads")
    heads: list[HeadContract] = []
    for index, raw in enumerate(raw_heads):
        item = _object(raw, f"heads[{index}]")
        _expect_keys(
            item,
            path=f"heads[{index}]",
            required=("name", "layer_index", "aggregation"),
        )
        heads.append(
            HeadContract(
                name=_string(item["name"], f"heads[{index}].name"),
                layer_index=_integer(item["layer_index"], f"heads[{index}].layer_index"),
                aggregation=_string(item["aggregation"], f"heads[{index}].aggregation"),
            )
        )
    expected = (
        ("mean_linear", -1),
        ("mean_layer_linear", -2),
        ("mean_rich_stats_residual", -1),
        ("multi_query_rich_stats", -1),
    )
    if len(heads) != 4 or tuple((head.name, head.layer_index) for head in heads) != expected:
        raise ProtocolError("protocol must contain exactly four approved heads and layer selections")
    return tuple(heads)


def _parse_training(value: object) -> TrainingContract:
    training = _object(value, "training")
    keys = (
        "seeds",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "loss",
        "checkpoint_metric",
        "metric_mode",
    )
    _expect_keys(training, path="training", required=keys)
    result = TrainingContract(
        seeds=tuple(
            _integer(item, "training.seeds[]")
            for item in _sequence(training["seeds"], "training.seeds")
        ),
        optimizer=_string(training["optimizer"], "training.optimizer"),
        learning_rate=_number(training["learning_rate"], "training.learning_rate"),
        weight_decay=_number(training["weight_decay"], "training.weight_decay"),
        batch_size=_integer(training["batch_size"], "training.batch_size"),
        max_epochs=_integer(training["max_epochs"], "training.max_epochs"),
        patience=_integer(training["patience"], "training.patience"),
        loss=_string(training["loss"], "training.loss"),
        checkpoint_metric=_string(training["checkpoint_metric"], "training.checkpoint_metric"),
        metric_mode=_string(training["metric_mode"], "training.metric_mode"),
    )
    if result.seeds != tuple(range(33, 43)):
        raise ProtocolError("training must use exactly seeds 33 through 42")
    if min(result.learning_rate, result.batch_size, result.max_epochs, result.patience) <= 0:
        raise ProtocolError("training numeric settings must be positive")
    if result.weight_decay < 0:
        raise ProtocolError("training.weight_decay must be non-negative")
    if result.checkpoint_metric != "validation_pearson" or result.metric_mode != "max":
        raise ProtocolError("checkpoint selection must maximize validation Pearson")
    return result


def _parse_statistics(value: object) -> StatisticsContract:
    statistics = _object(value, "statistics")
    keys = (
        "bootstrap_iterations",
        "bootstrap_seed",
        "confidence",
        "randomization_tail",
        "holm_order",
        "alpha",
        "minimum_seed_wins",
        "minimum_worst_seed_delta",
        "require_ci_above_zero",
    )
    _expect_keys(statistics, path="statistics", required=keys)
    result = StatisticsContract(
        bootstrap_iterations=_integer(
            statistics["bootstrap_iterations"], "statistics.bootstrap_iterations"
        ),
        bootstrap_seed=_integer(statistics["bootstrap_seed"], "statistics.bootstrap_seed"),
        confidence=_number(statistics["confidence"], "statistics.confidence"),
        randomization_tail=_string(
            statistics["randomization_tail"], "statistics.randomization_tail"
        ),
        holm_order=tuple(
            _string(item, "statistics.holm_order[]")
            for item in _sequence(statistics["holm_order"], "statistics.holm_order")
        ),
        alpha=_number(statistics["alpha"], "statistics.alpha"),
        minimum_seed_wins=_integer(
            statistics["minimum_seed_wins"], "statistics.minimum_seed_wins"
        ),
        minimum_worst_seed_delta=_number(
            statistics["minimum_worst_seed_delta"],
            "statistics.minimum_worst_seed_delta",
        ),
        require_ci_above_zero=_boolean(
            statistics["require_ci_above_zero"],
            "statistics.require_ci_above_zero",
        ),
    )
    if result.bootstrap_iterations != 10_000:
        raise ProtocolError("statistics must use 10,000 bootstrap iterations")
    if result.bootstrap_seed != 20260903:
        raise ProtocolError("statistics bootstrap seed must be 20260903")
    expected_order = (
        "mean_layer_linear",
        "mean_rich_stats_residual",
        "multi_query_rich_stats",
    )
    if result.holm_order != expected_order:
        raise ProtocolError("Holm order must contain the three approved candidates")
    if (
        result.confidence != 0.95
        or result.randomization_tail != "greater"
        or result.alpha != 0.05
        or result.minimum_seed_wins != 8
        or result.minimum_worst_seed_delta != -0.01
        or not result.require_ci_above_zero
    ):
        raise ProtocolError("statistics decision thresholds do not match the approved protocol")
    return result


def load_study_protocol(path: Path) -> StudyProtocol:
    """Load and validate the complete confirmatory protocol without defaults."""

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"could not read protocol JSON: {path}") from error
    payload = _object(raw, "protocol")
    _expect_keys(
        payload,
        path="protocol",
        required=(
            "schema_version",
            "study_id",
            "status",
            "datasets",
            "encoder",
            "preprocessing",
            "heads",
            "training",
            "statistics",
        ),
    )
    schema_version = _integer(payload["schema_version"], "schema_version")
    if schema_version != 1:
        raise ProtocolError("schema_version must be 1")
    status = _string(payload["status"], "status")
    if status not in {"draft", "sealed"}:
        raise ProtocolError("protocol status must be draft or sealed")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return StudyProtocol(
        schema_version=schema_version,
        study_id=_string(payload["study_id"], "study_id"),
        status=status,
        datasets=_parse_datasets(payload["datasets"]),
        encoder=_parse_encoder(payload["encoder"]),
        preprocessing=_parse_preprocessing(payload["preprocessing"]),
        heads=_parse_heads(payload["heads"]),
        training=_parse_training(payload["training"]),
        statistics=_parse_statistics(payload["statistics"]),
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )
