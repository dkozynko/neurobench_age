"""Strict exploratory inference metadata for the capacity--data extension."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class CapacityExploratoryInferenceError(ValueError):
    """Raised when exploratory inference metadata is missing or altered."""


_FIELDS = {
    "schema_version",
    "status",
    "scope",
    "alpha",
    "tail",
    "zero_deltas",
    "cell_family_order",
    "contrast_family_order",
    "decision_rule",
}
_CELL_HEADS = (
    "mean_rich_stats_residual",
    "mean_mlp_residual_matched(hidden_dim=4)",
)
_SIZES = (200, 400, 800)
_CONTRASTS = (
    "endpoint_800_minus_200",
    "adjacent_400_minus_200",
    "adjacent_800_minus_400",
)
_DEFAULT_PAYLOAD: dict[str, Any] = {
    "schema_version": 1,
    "status": "exploratory",
    "scope": "capacity_data_regime_secondary",
    "alpha": 0.05,
    "tail": "greater",
    "zero_deltas": "include_as_nonnegative_exceedance",
    "cell_family_order": [
        f"{head}@{size}"
        for head in _CELL_HEADS
        for size in _SIZES
    ],
    "contrast_family_order": [
        f"{head}@{contrast}"
        for head in _CELL_HEADS
        for contrast in _CONTRASTS
    ],
    "decision_rule": "exploratory_p_values_do_not_change_sealed_stable_improvement_decision",
}


@dataclass(frozen=True)
class CapacityExploratoryInference:
    schema_version: int
    status: str
    scope: str
    alpha: float
    tail: str
    zero_deltas: str
    cell_family_order: tuple[str, ...]
    contrast_family_order: tuple[str, ...]
    decision_rule: str
    sha256: str


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(payload) - _FIELDS
    missing = _FIELDS - set(payload)
    if unknown or missing:
        raise CapacityExploratoryInferenceError(
            f"inference config fields differ: unknown={sorted(unknown)} missing={sorted(missing)}"
        )
    if payload["schema_version"] != 1 or payload["status"] != "exploratory":
        raise CapacityExploratoryInferenceError("inference config version/status is invalid")
    if payload["scope"] != "capacity_data_regime_secondary":
        raise CapacityExploratoryInferenceError("inference config scope is invalid")
    alpha = payload["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise CapacityExploratoryInferenceError("inference alpha must be numeric")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha != 0.05:
        raise CapacityExploratoryInferenceError("inference alpha must be exactly 0.05")
    if payload["tail"] != "greater":
        raise CapacityExploratoryInferenceError("only the predeclared greater tail is supported")
    if payload["zero_deltas"] != "include_as_nonnegative_exceedance":
        raise CapacityExploratoryInferenceError("zero-delta handling is invalid")
    expected_cells = tuple(_DEFAULT_PAYLOAD["cell_family_order"])
    expected_contrasts = tuple(_DEFAULT_PAYLOAD["contrast_family_order"])
    for field, expected in (
        ("cell_family_order", expected_cells),
        ("contrast_family_order", expected_contrasts),
    ):
        value = payload[field]
        if not isinstance(value, list) or tuple(value) != expected:
            raise CapacityExploratoryInferenceError(f"{field} is not the exact declared order")
    if payload["decision_rule"] != _DEFAULT_PAYLOAD["decision_rule"]:
        raise CapacityExploratoryInferenceError("inference decision rule is invalid")
    return {
        **dict(payload),
        "alpha": alpha,
    }


def _from_payload(payload: Mapping[str, Any]) -> CapacityExploratoryInference:
    validated = _validate_payload(payload)
    return CapacityExploratoryInference(
        schema_version=int(validated["schema_version"]),
        status=str(validated["status"]),
        scope=str(validated["scope"]),
        alpha=float(validated["alpha"]),
        tail=str(validated["tail"]),
        zero_deltas=str(validated["zero_deltas"]),
        cell_family_order=tuple(validated["cell_family_order"]),
        contrast_family_order=tuple(validated["contrast_family_order"]),
        decision_rule=str(validated["decision_rule"]),
        sha256=_canonical_sha256(validated),
    )


def load_capacity_exploratory_inference(path: Path) -> CapacityExploratoryInference:
    """Load and validate the exact exploratory inference specification."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityExploratoryInferenceError(
            f"could not read exploratory inference config: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise CapacityExploratoryInferenceError("exploratory inference config must be an object")
    return _from_payload(payload)


def default_capacity_exploratory_inference() -> CapacityExploratoryInference:
    """Return the repository-declared default for direct library callers."""

    return _from_payload(_DEFAULT_PAYLOAD)
