"""Strict JSON, RFC 8785 canonicalization, and schema helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import jsonschema
import rfc8785


class StrictJsonError(ValueError):
    """Raised when a research artifact violates strict JSON rules."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise StrictJsonError(f"non-finite JSON number: {value}")


def load_json_strict(source: Path | str | bytes | bytearray) -> Any:
    """Load JSON while rejecting duplicate keys and non-standard numbers."""

    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = bytes(source)
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except StrictJsonError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrictJsonError("invalid UTF-8 JSON") from error


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise StrictJsonError("non-finite number cannot be canonicalized")
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_nonfinite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nonfinite(nested)


def _without_fields(value: Any, excluded: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_fields(nested, excluded)
            for key, nested in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [_without_fields(item, excluded) for item in value]
    if isinstance(value, tuple):
        return [_without_fields(item, excluded) for item in value]
    return value


def canonical_json_bytes(
    value: Any, *, exclude_fields: Iterable[str] = ()
) -> bytes:
    """Serialize a JSON-compatible value using RFC 8785 canonical JSON."""

    _reject_nonfinite(value)
    normalized = _without_fields(value, set(exclude_fields))
    try:
        return rfc8785.dumps(normalized)
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as error:
        raise StrictJsonError("value cannot be canonically serialized") from error


def canonical_sha256(
    value: Any, *, exclude_fields: Iterable[str] = ()
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, exclude_fields=exclude_fields)
    ).hexdigest()


def validate_schema(value: Any, schema_path: Path | str) -> None:
    schema = load_json_strict(schema_path)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    except jsonschema.SchemaError:
        raise


def reject_target_fields(
    value: Any,
    *,
    forbidden_fields: Iterable[str] = (
        "age",
        "age_years",
        "target",
        "prediction",
        "pearson",
        "mae",
        "rmse",
        "r_squared",
        "metrics",
    ),
) -> None:
    """Reject target-bearing field names anywhere in an artifact payload."""

    forbidden = {field.casefold() for field in forbidden_fields}

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, nested in node.items():
                if str(key).casefold() in forbidden:
                    raise StrictJsonError(
                        f"target-bearing field is forbidden in target-free artifact: {key}"
                    )
                visit(nested)
        elif isinstance(node, (list, tuple)):
            for nested in node:
                visit(nested)

    visit(value)


def write_create_only_json(
    path: Path | str,
    value: Mapping[str, Any],
    schema_path: Path | str,
    hash_field: str,
) -> dict[str, Any]:
    """Validate, hash, and atomically create one immutable JSON artifact."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    body = dict(value)
    body[hash_field] = canonical_sha256(body, exclude_fields=(hash_field,))
    validate_schema(body, schema_path)
    encoded = canonical_json_bytes(body)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(encoded)
        handle.write(b"\n")
    return body
