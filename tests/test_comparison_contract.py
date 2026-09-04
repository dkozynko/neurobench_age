from __future__ import annotations

import pytest

from neurobench_age.analysis.comparison import (
    ComparisonContractError,
    ComparisonRecord,
    match_exact_records,
)


FIELDS = ("dataset_manifest", "split_fingerprint", "encoder_checkpoint")


def _record(
    seed: int,
    *,
    subjects: tuple[str, ...] = ("s-1", "s-2"),
    targets: tuple[float, ...] = (10.0, 20.0),
    **provenance: object,
) -> ComparisonRecord:
    values = {
        "dataset_manifest": "dataset-sha",
        "split_fingerprint": "split-sha",
        "encoder_checkpoint": "brain-bzh/reve-base",
        **provenance,
    }
    return ComparisonRecord(
        seed=seed,
        provenance=values,
        subject_ids=subjects,
        true_targets=targets,
        payload={"seed": seed},
    )


def test_match_exact_records_requires_identical_seed_inventory() -> None:
    baseline = [_record(33), _record(34)]
    candidate = [_record(33)]

    with pytest.raises(ComparisonContractError) as raised:
        match_exact_records(baseline, candidate, provenance_fields=FIELDS)

    assert raised.value.mismatches == (
        {
            "field": "seed_inventory",
            "baseline": [33, 34],
            "candidate": [33],
        },
    )


def test_match_exact_records_reports_every_provenance_mismatch() -> None:
    baseline = [_record(33)]
    candidate = [
        _record(
            33,
            dataset_manifest="other-dataset",
            split_fingerprint="other-split",
        )
    ]

    with pytest.raises(ComparisonContractError) as raised:
        match_exact_records(baseline, candidate, provenance_fields=FIELDS)

    assert [item["field"] for item in raised.value.mismatches] == [
        "dataset_manifest",
        "split_fingerprint",
    ]


def test_match_exact_records_rejects_missing_required_provenance() -> None:
    baseline = [_record(33)]
    candidate = [_record(33)]
    baseline[0].provenance.pop("encoder_checkpoint")
    candidate[0].provenance.pop("encoder_checkpoint")

    with pytest.raises(ComparisonContractError) as raised:
        match_exact_records(baseline, candidate, provenance_fields=FIELDS)

    assert raised.value.mismatches[0]["field"] == "encoder_checkpoint"


def test_match_exact_records_requires_ordered_subjects_and_targets() -> None:
    baseline = [_record(33)]
    candidate = [
        _record(
            33,
            subjects=("s-2", "s-1"),
            targets=(20.0, 10.0),
        )
    ]

    with pytest.raises(ComparisonContractError) as raised:
        match_exact_records(baseline, candidate, provenance_fields=FIELDS)

    assert [item["field"] for item in raised.value.mismatches] == [
        "subject_ids",
        "true_targets",
    ]


def test_match_exact_records_rejects_duplicate_seed() -> None:
    with pytest.raises(ComparisonContractError, match="duplicate"):
        match_exact_records(
            [_record(33), _record(33)],
            [_record(33)],
            provenance_fields=FIELDS,
        )


def test_match_exact_records_returns_seed_sorted_pairs() -> None:
    pairs = match_exact_records(
        [_record(34), _record(33)],
        [_record(33), _record(34)],
        provenance_fields=FIELDS,
    )

    assert [pair.seed for pair in pairs] == [33, 34]
