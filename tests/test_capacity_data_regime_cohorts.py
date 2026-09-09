from __future__ import annotations

from pathlib import Path

import pytest

from neurobench_age.research.capacity_data_regime import (
    CapacityDataRegimeProtocolError,
    build_nested_training_cohorts,
    canonical_cache_manifest,
    cohort_sha256,
    load_capacity_data_regime_protocol,
    priority_digest,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_capacity_data_regime_protocol(
    ROOT / "configs/research/capacity_data_regime.json", repository_root=ROOT
)


def _subject_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for release in PROTOCOL.eligible_training_releases:
        for index in range(1, 101):
            rows.append(
                {
                    "subject_id": f"{release}-S{index:03d}",
                    "release": release,
                    "split": "train",
                }
            )
    rows.extend(
        [
            {"subject_id": "R5-S001", "release": "R5", "split": "test"},
            {"subject_id": "R8-S001", "release": "R8", "split": "validation"},
        ]
    )
    return rows


def test_nested_cohorts_are_release_balanced_and_nested() -> None:
    cohorts = build_nested_training_cohorts(_subject_rows(), protocol=PROTOCOL)

    assert {size: len(subjects) for size, subjects in cohorts.items()} == {
        200: 200,
        400: 400,
        800: 800,
    }
    for size in (200, 400, 800):
        counts = {
            release: sum(subject_id.startswith(f"{release}-") for subject_id in cohorts[size])
            for release in PROTOCOL.eligible_training_releases
        }
        assert set(counts.values()) == {size // 8}
    assert set(cohorts[200]).issubset(cohorts[400])
    assert set(cohorts[400]).issubset(cohorts[800])
    assert not any(subject_id.startswith("R5-") for subject_id in cohorts[800])
    assert not any(subject_id.startswith("R8-") for subject_id in cohorts[800])


def test_nested_cohorts_are_independent_of_source_row_order() -> None:
    first = build_nested_training_cohorts(_subject_rows(), protocol=PROTOCOL)
    second = build_nested_training_cohorts(list(reversed(_subject_rows())), protocol=PROTOCOL)

    assert first == second
    assert priority_digest("R1-S001", salt=PROTOCOL.priority_salt) != priority_digest(
        "R1-S002", salt=PROTOCOL.priority_salt
    )


def test_nested_cohort_rejects_duplicate_or_invalid_training_subjects() -> None:
    duplicate = _subject_rows() + [_subject_rows()[0]]
    with pytest.raises(CapacityDataRegimeProtocolError, match="duplicate"):
        build_nested_training_cohorts(duplicate, protocol=PROTOCOL)

    invalid = _subject_rows()
    invalid[0] = {"subject_id": "R5-S999", "release": "R5", "split": "train"}
    with pytest.raises(CapacityDataRegimeProtocolError, match="R5"):
        build_nested_training_cohorts(invalid, protocol=PROTOCOL)


def _cache_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject_id, split in (("sub-b", "validation"), ("sub-a", "train")):
        for recording_window_key in ("recording-02/window-001", "recording-01/window-002"):
            for layer_index in (-1, -2):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "split": split,
                        "recording_window_key": recording_window_key,
                        "cache_key": f"cache-{subject_id}-{recording_window_key}-{layer_index}",
                        "layer_index": layer_index,
                        "shape": [3, 4, 5],
                        "dtype": "torch.float32",
                        "payload_sha256": "a" * 64,
                    }
                )
    return rows


def test_cache_manifest_uses_exact_canonical_order_and_includes_validation() -> None:
    canonical, digest = canonical_cache_manifest(list(reversed(_cache_rows())))

    assert len(canonical) == 8
    assert {row["split"] for row in canonical} == {"train", "validation"}
    assert [row["split"] for row in canonical[:4]] == ["train"] * 4
    train_rows = [row for row in canonical if row["split"] == "train"]
    assert [row["layer_index"] for row in train_rows[:2]] == [-2, -1]
    assert canonical[0]["subject_id"] == "sub-a"
    assert canonical[-1]["subject_id"] == "sub-b"
    assert len(digest) == 64


def test_cache_manifest_rejects_missing_duplicate_and_unexpected_rows() -> None:
    rows = _cache_rows()
    with pytest.raises(CapacityDataRegimeProtocolError, match="duplicate"):
        canonical_cache_manifest(rows + [rows[0]])

    incomplete = rows[:-1]
    with pytest.raises(CapacityDataRegimeProtocolError, match="expected cache rows"):
        canonical_cache_manifest(incomplete, expected_rows=rows)

    unexpected = list(rows)
    unexpected[0] = dict(unexpected[0], layer_index=-3)
    with pytest.raises(CapacityDataRegimeProtocolError, match="layer"):
        canonical_cache_manifest(unexpected)


def test_cohort_hash_binds_source_and_cache_digests() -> None:
    subject_ids = ("R1-S001", "R1-S002")
    first = cohort_sha256(
        extension_version="reve_age_capacity_data_regime_v1",
        split_name="train-200",
        subject_ids=subject_ids,
        source_manifest_sha256="b" * 64,
        representation_cache_sha256="c" * 64,
    )
    changed = cohort_sha256(
        extension_version="reve_age_capacity_data_regime_v1",
        split_name="train-200",
        subject_ids=subject_ids,
        source_manifest_sha256="d" * 64,
        representation_cache_sha256="c" * 64,
    )

    assert first != changed
    assert len(first) == 64
