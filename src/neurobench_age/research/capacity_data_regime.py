"""Strict protocol and deterministic inputs for the capacity--data extension."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class CapacityDataRegimeProtocolError(ValueError):
    """Raised when the extension protocol is missing, altered, or invalid."""


EXPECTED_TRAINING_PROTOCOL_PATH = Path(
    "configs/research/neuralbench_frozen_probe_training.json"
)
EXPECTED_TRAINING_PROTOCOL_FILE_SHA256 = (
    "721f745080e262d7813d2a67056643afae89dbf1b259524e6d5d3951bba3b4d6"
)
EXPECTED_REPRESENTATION_PROTOCOL_PATH = Path(
    "configs/research/external_frozen_probe.json"
)
EXPECTED_REPRESENTATION_PROTOCOL_FILE_SHA256 = (
    "654923b15f9d4f4ccec0eb02637898b289bd8879326c71a96e5f6d88e698c60f"
)
EXPECTED_HEAD_NAMES = (
    "mean_linear",
    "mean_rich_stats_residual",
    "mean_mlp_residual_matched(hidden_dim=4)",
)
EXPECTED_TRAINING_RELEASES = (
    "R1",
    "R2",
    "R3",
    "R4",
    "R6",
    "R7",
    "R9",
    "R10",
)


@dataclass(frozen=True)
class CapacityHeadSpec:
    name: str
    layer_index: int
    aggregation: str
    hidden_dim: int | None


@dataclass(frozen=True)
class CapacityDataRegimeProtocol:
    schema_version: int
    status: str
    extension_id: str
    training_sizes: tuple[int, ...]
    future_endpoints: tuple[str, ...]
    seeds: tuple[int, ...]
    eligible_training_releases: tuple[str, ...]
    validation_release: str
    test_release: str
    priority_salt: str
    heads: tuple[CapacityHeadSpec, ...]
    training_protocol_path: Path
    training_protocol_file_sha256: str
    representation_protocol_path: Path
    representation_protocol_file_sha256: str
    bootstrap_iterations: int
    bootstrap_seed: int
    bootstrap_confidence: float
    minimum_valid_bootstrap_replicates: int
    external_primary_subject_count: int
    expected_run_count: int
    expected_prediction_count: int
    allow_extrapolation: bool
    free_space_floor_bytes: int
    free_space_cache_fraction: float
    free_space_output_multiplier: int
    sha256: str

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(head.name for head in self.heads)

    @property
    def training_protocol_source(self) -> str:
        return self.training_protocol_path.as_posix()

    @property
    def representation_protocol_source(self) -> str:
        return self.representation_protocol_path.as_posix()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CapacityDataRegimeProtocolError(f"{path} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise CapacityDataRegimeProtocolError(
            f"{path} has unknown fields={unknown} missing fields={missing}"
        )


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CapacityDataRegimeProtocolError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CapacityDataRegimeProtocolError(f"{path} must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapacityDataRegimeProtocolError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CapacityDataRegimeProtocolError(f"{path} must be finite")
    return result


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise CapacityDataRegimeProtocolError(f"{path} must be boolean")
    return value


def _sha256_file(path: Path, field: str) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise CapacityDataRegimeProtocolError(
            f"could not read pinned {field}: {path}"
        ) from error
    return digest


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def priority_digest(subject_id: str, *, salt: str) -> str:
    """Return the exact UTF-8 priority digest used for nested sampling."""

    if not isinstance(subject_id, str) or not subject_id:
        raise CapacityDataRegimeProtocolError("subject_id must be a non-empty string")
    try:
        salt_bytes = salt.encode("ascii")
    except UnicodeEncodeError as error:
        raise CapacityDataRegimeProtocolError("priority salt must be ASCII") from error
    return hashlib.sha256(salt_bytes + b"\x00" + subject_id.encode("utf-8")).hexdigest()


def build_nested_training_cohorts(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: CapacityDataRegimeProtocol,
) -> dict[int, tuple[str, ...]]:
    """Build the three release-balanced, deterministic nested train cohorts."""

    seen_subjects: set[str] = set()
    by_release: dict[str, list[str]] = {
        release: [] for release in protocol.eligible_training_releases
    }
    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"cohort rows[{index}]")
        subject_id = _string(row.get("subject_id"), f"cohort rows[{index}].subject_id")
        release = _string(row.get("release"), f"cohort rows[{index}].release")
        split = _string(row.get("split"), f"cohort rows[{index}].split")
        if subject_id in seen_subjects:
            raise CapacityDataRegimeProtocolError(
                f"duplicate subject_id in cohort source: {subject_id}"
            )
        seen_subjects.add(subject_id)
        if release in {protocol.test_release, protocol.validation_release}:
            if release == protocol.test_release and split != "test":
                raise CapacityDataRegimeProtocolError(
                    f"{protocol.test_release} must remain the test split"
                )
            if release == protocol.validation_release and split != "validation":
                raise CapacityDataRegimeProtocolError(
                    f"{protocol.validation_release} must remain the validation split"
                )
            continue
        if release not in by_release:
            raise CapacityDataRegimeProtocolError(
                f"unexpected release in cohort source: {release}"
            )
        if split != "train":
            raise CapacityDataRegimeProtocolError(
                f"eligible release {release} must use the train split"
            )
        by_release[release].append(subject_id)

    for release, subject_ids in by_release.items():
        if len(subject_ids) != 100:
            raise CapacityDataRegimeProtocolError(
                f"release {release} must contribute exactly 100 training subjects"
            )
        subject_ids.sort(
            key=lambda subject_id: (
                priority_digest(subject_id, salt=protocol.priority_salt),
                subject_id.encode("utf-8"),
            )
        )

    cohorts: dict[int, tuple[str, ...]] = {}
    for size in protocol.training_sizes:
        per_release = size // len(protocol.eligible_training_releases)
        if per_release * len(protocol.eligible_training_releases) != size:
            raise CapacityDataRegimeProtocolError(
                "training size must divide evenly across eligible releases"
            )
        cohorts[size] = tuple(
            subject_id
            for release in protocol.eligible_training_releases
            for subject_id in by_release[release][:per_release]
        )
    return cohorts


def canonical_cache_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Canonicalize and hash every train/validation cache row."""

    required = {
        "subject_id",
        "split",
        "recording_window_key",
        "cache_key",
        "layer_index",
        "shape",
        "dtype",
        "payload_sha256",
    }
    normalized: list[dict[str, Any]] = []
    row_keys: set[tuple[str, str, str, int, str]] = set()
    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"cache rows[{index}]")
        _exact_fields(row, required, f"cache rows[{index}]")
        subject_id = _string(row["subject_id"], f"cache rows[{index}].subject_id")
        split = _string(row["split"], f"cache rows[{index}].split")
        if split not in {"train", "validation"}:
            raise CapacityDataRegimeProtocolError(
                f"cache rows[{index}] has invalid split"
            )
        recording_window_key = _string(
            row["recording_window_key"],
            f"cache rows[{index}].recording_window_key",
        )
        cache_key = _string(row["cache_key"], f"cache rows[{index}].cache_key")
        layer_index = _integer(row["layer_index"], f"cache rows[{index}].layer_index")
        if layer_index not in {-2, -1}:
            raise CapacityDataRegimeProtocolError(
                f"cache rows[{index}] has invalid layer index"
            )
        shape = row["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            raise CapacityDataRegimeProtocolError(
                f"cache rows[{index}] has invalid tensor shape"
            )
        dtype = _string(row["dtype"], f"cache rows[{index}].dtype")
        payload_sha256 = _string(
            row["payload_sha256"], f"cache rows[{index}].payload_sha256"
        )
        if not _is_sha256(payload_sha256):
            raise CapacityDataRegimeProtocolError(
                f"cache rows[{index}] payload SHA-256 is invalid"
            )
        row_key = (split, subject_id, recording_window_key, layer_index, cache_key)
        if row_key in row_keys:
            raise CapacityDataRegimeProtocolError("duplicate cache row")
        row_keys.add(row_key)
        normalized.append(
            {
                "subject_id": subject_id,
                "split": split,
                "split_order": 0 if split == "train" else 1,
                "recording_window_key": recording_window_key,
                "cache_key": cache_key,
                "layer_index": layer_index,
                "shape": list(shape),
                "dtype": dtype,
                "payload_sha256": payload_sha256,
            }
        )

    if expected_rows is not None:
        expected_keys = set()
        for index, expected in enumerate(expected_rows):
            value = _object(expected, f"expected cache rows[{index}]")
            try:
                expected_keys.add(
                    (
                        str(value["split"]),
                        str(value["subject_id"]),
                        str(value["recording_window_key"]),
                        int(value["layer_index"]),
                        str(value["cache_key"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise CapacityDataRegimeProtocolError(
                    "expected cache rows are malformed"
                ) from error
        if row_keys != expected_keys:
            raise CapacityDataRegimeProtocolError(
                "actual and expected cache rows differ"
            )

    normalized.sort(
        key=lambda row: (
            row["split_order"],
            row["subject_id"].encode("utf-8"),
            row["recording_window_key"].encode("utf-8"),
            row["layer_index"],
            row["cache_key"].encode("utf-8"),
        )
    )
    canonical = tuple(normalized)
    digest = _canonical_sha256({"schema_version": 1, "rows": canonical})
    return canonical, digest


def cohort_sha256(
    *,
    extension_version: str,
    split_name: str,
    subject_ids: Sequence[str],
    source_manifest_sha256: str,
    representation_cache_sha256: str,
) -> str:
    """Hash an ordered cohort and both source identities."""

    if not _is_sha256(source_manifest_sha256) or not _is_sha256(
        representation_cache_sha256
    ):
        raise CapacityDataRegimeProtocolError("cohort source hashes must be SHA-256")
    if any(not isinstance(subject_id, str) or not subject_id for subject_id in subject_ids):
        raise CapacityDataRegimeProtocolError("cohort subject IDs must be non-empty strings")
    return _canonical_sha256(
        {
            "extension_version": extension_version,
            "split_name": split_name,
            "subject_ids": list(subject_ids),
            "source_manifest_sha256": source_manifest_sha256,
            "representation_cache_sha256": representation_cache_sha256,
        }
    )


def _parse_heads(raw: object) -> tuple[CapacityHeadSpec, ...]:
    if not isinstance(raw, list):
        raise CapacityDataRegimeProtocolError("heads must be an array")
    expected_fields = {"name", "layer_index", "aggregation", "hidden_dim"}
    parsed: list[CapacityHeadSpec] = []
    for index, value in enumerate(raw):
        head = _object(value, f"heads[{index}]")
        _exact_fields(head, expected_fields, f"heads[{index}]")
        hidden_dim = head["hidden_dim"]
        if hidden_dim is not None:
            hidden_dim = _integer(hidden_dim, f"heads[{index}].hidden_dim")
        parsed.append(
            CapacityHeadSpec(
                name=_string(head["name"], f"heads[{index}].name"),
                layer_index=_integer(
                    head["layer_index"], f"heads[{index}].layer_index"
                ),
                aggregation=_string(
                    head["aggregation"], f"heads[{index}].aggregation"
                ),
                hidden_dim=hidden_dim,
            )
        )
    return tuple(parsed)


def load_capacity_data_regime_protocol(
    path: Path, *, repository_root: Path | None = None
) -> CapacityDataRegimeProtocol:
    """Load and fail closed against the frozen extension contract."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityDataRegimeProtocolError(
            f"could not read capacity-data-regime protocol: {path}"
        ) from error

    root = _object(payload, "capacity-data-regime protocol")
    _exact_fields(
        root,
        {
            "schema_version",
            "status",
            "extension_id",
            "training_sizes",
            "future_endpoints",
            "seeds",
            "eligible_training_releases",
            "validation_release",
            "test_release",
            "priority_salt",
            "heads",
            "training_protocol",
            "representation_protocol",
            "external",
            "preflight",
            "bootstrap",
        },
        "capacity-data-regime protocol",
    )

    def integer_tuple(value: object, field: str) -> tuple[int, ...]:
        if not isinstance(value, list):
            raise CapacityDataRegimeProtocolError(f"{field} must be an array")
        return tuple(_integer(item, f"{field}[]") for item in value)

    def string_tuple(value: object, field: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise CapacityDataRegimeProtocolError(f"{field} must be an array")
        return tuple(_string(item, f"{field}[]") for item in value)

    training_protocol = _object(root["training_protocol"], "training_protocol")
    _exact_fields(training_protocol, {"path", "sha256"}, "training_protocol")
    representation_protocol = _object(
        root["representation_protocol"], "representation_protocol"
    )
    _exact_fields(
        representation_protocol,
        {"path", "sha256"},
        "representation_protocol",
    )
    external = _object(root["external"], "external")
    _exact_fields(
        external,
        {
            "primary_subject_count",
            "expected_run_count",
            "expected_prediction_count",
            "allow_extrapolation",
        },
        "external",
    )
    preflight = _object(root["preflight"], "preflight")
    _exact_fields(
        preflight,
        {
            "free_space_floor_bytes",
            "free_space_cache_fraction",
            "free_space_output_multiplier",
        },
        "preflight",
    )
    bootstrap = _object(root["bootstrap"], "bootstrap")
    _exact_fields(
        bootstrap,
        {"iterations", "seed", "confidence", "minimum_valid_replicates"},
        "bootstrap",
    )

    training_sizes = integer_tuple(root["training_sizes"], "training_sizes")
    future_endpoints = string_tuple(root["future_endpoints"], "future_endpoints")
    seeds = integer_tuple(root["seeds"], "seeds")
    eligible_releases = string_tuple(
        root["eligible_training_releases"], "eligible_training_releases"
    )
    training_protocol_path = Path(
        _string(training_protocol["path"], "training_protocol.path")
    )
    training_protocol_file_sha256 = _string(
        training_protocol["sha256"], "training_protocol.sha256"
    )
    representation_protocol_path = Path(
        _string(representation_protocol["path"], "representation_protocol.path")
    )
    representation_protocol_file_sha256 = _string(
        representation_protocol["sha256"], "representation_protocol.sha256"
    )

    result = CapacityDataRegimeProtocol(
        schema_version=_integer(root["schema_version"], "schema_version"),
        status=_string(root["status"], "status"),
        extension_id=_string(root["extension_id"], "extension_id"),
        training_sizes=training_sizes,
        future_endpoints=future_endpoints,
        seeds=seeds,
        eligible_training_releases=eligible_releases,
        validation_release=_string(root["validation_release"], "validation_release"),
        test_release=_string(root["test_release"], "test_release"),
        priority_salt=_string(root["priority_salt"], "priority_salt"),
        heads=_parse_heads(root["heads"]),
        training_protocol_path=training_protocol_path,
        training_protocol_file_sha256=training_protocol_file_sha256,
        representation_protocol_path=representation_protocol_path,
        representation_protocol_file_sha256=representation_protocol_file_sha256,
        bootstrap_iterations=_integer(bootstrap["iterations"], "bootstrap.iterations"),
        bootstrap_seed=_integer(bootstrap["seed"], "bootstrap.seed"),
        bootstrap_confidence=_number(
            bootstrap["confidence"], "bootstrap.confidence"
        ),
        minimum_valid_bootstrap_replicates=_integer(
            bootstrap["minimum_valid_replicates"],
            "bootstrap.minimum_valid_replicates",
        ),
        external_primary_subject_count=_integer(
            external["primary_subject_count"], "external.primary_subject_count"
        ),
        expected_run_count=_integer(
            external["expected_run_count"], "external.expected_run_count"
        ),
        expected_prediction_count=_integer(
            external["expected_prediction_count"],
            "external.expected_prediction_count",
        ),
        allow_extrapolation=_boolean(
            external["allow_extrapolation"], "external.allow_extrapolation"
        ),
        free_space_floor_bytes=_integer(
            preflight["free_space_floor_bytes"],
            "preflight.free_space_floor_bytes",
        ),
        free_space_cache_fraction=_number(
            preflight["free_space_cache_fraction"],
            "preflight.free_space_cache_fraction",
        ),
        free_space_output_multiplier=_integer(
            preflight["free_space_output_multiplier"],
            "preflight.free_space_output_multiplier",
        ),
        sha256=_canonical_sha256(root),
    )

    if result.schema_version != 1 or result.status != "final":
        raise CapacityDataRegimeProtocolError(
            "capacity-data-regime protocol must be schema 1 with final status"
        )
    if result.extension_id != "reve_age_capacity_data_regime_v1":
        raise CapacityDataRegimeProtocolError("extension_id is not approved")
    if result.training_sizes != (200, 400, 800):
        raise CapacityDataRegimeProtocolError(
            "training sizes must be exactly 200, 400, and 800"
        )
    if result.future_endpoints != ("all_available",):
        raise CapacityDataRegimeProtocolError(
            "future endpoints must contain only all_available"
        )
    if result.seeds != tuple(range(33, 43)):
        raise CapacityDataRegimeProtocolError(
            "extension must use exactly seeds 33 through 42"
        )
    if result.eligible_training_releases != EXPECTED_TRAINING_RELEASES:
        raise CapacityDataRegimeProtocolError(
            "eligible training releases do not match the predeclared eight-release set"
        )
    if result.validation_release != "R8" or result.test_release != "R5":
        raise CapacityDataRegimeProtocolError(
            "validation release must be R8 and test release must be R5"
        )
    if result.priority_salt != "capacity-data-regime-v1":
        raise CapacityDataRegimeProtocolError("priority salt is not approved")
    if result.head_names != EXPECTED_HEAD_NAMES:
        raise CapacityDataRegimeProtocolError(
            "extension heads do not match the exact allowlist"
        )
    for head in result.heads[:2]:
        if head.layer_index != -1 or head.hidden_dim is not None:
            raise CapacityDataRegimeProtocolError(
                "linear and rich-statistics heads must use final layer without hidden_dim"
            )
    matched = result.heads[2]
    if (
        matched.layer_index != -1
        or matched.aggregation != "mean_mlp_residual"
        or matched.hidden_dim != 4
    ):
        raise CapacityDataRegimeProtocolError(
            "matched MLP must use final-layer mean pooling with hidden_dim=4"
        )
    if result.training_protocol_path != EXPECTED_TRAINING_PROTOCOL_PATH:
        raise CapacityDataRegimeProtocolError(
            "training protocol path must be the pinned NeuralBench training contract"
        )
    if result.training_protocol_file_sha256 != EXPECTED_TRAINING_PROTOCOL_FILE_SHA256:
        raise CapacityDataRegimeProtocolError(
            "training protocol file digest is not approved"
        )
    if result.representation_protocol_path != EXPECTED_REPRESENTATION_PROTOCOL_PATH:
        raise CapacityDataRegimeProtocolError(
            "representation protocol path must be the pinned external contract"
        )
    if (
        result.representation_protocol_file_sha256
        != EXPECTED_REPRESENTATION_PROTOCOL_FILE_SHA256
    ):
        raise CapacityDataRegimeProtocolError(
            "representation protocol file digest is not approved"
        )
    if result.bootstrap_iterations != 10_000 or result.bootstrap_seed != 20260909:
        raise CapacityDataRegimeProtocolError(
            "bootstrap must use 10,000 iterations and seed 20260909"
        )
    if result.bootstrap_confidence != 0.95:
        raise CapacityDataRegimeProtocolError("bootstrap confidence must be 0.95")
    if result.minimum_valid_bootstrap_replicates != 9_950:
        raise CapacityDataRegimeProtocolError(
            "bootstrap must retain at least 9,950 valid replicates"
        )
    if (
        result.external_primary_subject_count != 75
        or result.expected_run_count != 90
        or result.expected_prediction_count != 6_750
        or result.allow_extrapolation
    ):
        raise CapacityDataRegimeProtocolError(
            "external extension inventory counts or extrapolation policy are invalid"
        )
    if (
        result.free_space_floor_bytes != 12 * 1024**3
        or result.free_space_cache_fraction != 0.25
        or result.free_space_output_multiplier != 2
    ):
        raise CapacityDataRegimeProtocolError("preflight storage contract is invalid")

    root_path = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[3]
    training_file = root_path / result.training_protocol_path
    representation_file = root_path / result.representation_protocol_path
    if _sha256_file(training_file, "training protocol") != result.training_protocol_file_sha256:
        raise CapacityDataRegimeProtocolError(
            "pinned training protocol file content does not match its declared digest"
        )
    if _sha256_file(representation_file, "representation protocol") != result.representation_protocol_file_sha256:
        raise CapacityDataRegimeProtocolError(
            "pinned representation protocol file content does not match its declared digest"
        )
    return result
