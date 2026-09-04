"""Fail-closed contracts for paired experimental comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EXACT_RUN_PROVENANCE_FIELDS = (
    "task",
    "dataset_manifest",
    "split_fingerprint",
    "source_tree_sha256",
    "protocol_digest",
    "comparison_config_hash",
    "comparison_factor_keys",
    "encoder_checkpoint",
    "deterministic_policy",
    "deterministic_policy_satisfied",
    "deterministic_settings",
    "evaluation_mode",
    "test_access",
)


class ComparisonContractError(ValueError):
    """Raised when two run groups are not exactly comparable."""

    def __init__(self, mismatches: Sequence[Mapping[str, Any]]) -> None:
        self.mismatches = tuple(dict(item) for item in mismatches)
        labels = {
            "comparison_config_hash": "comparison config",
            "comparison_factor_keys": "comparison factor declaration",
            "protocol_digest": "protocol digest",
            "source_tree_sha256": "source tree",
            "subject_ids": "exactly the same ordered subject IDs",
            "true_targets": "identical true targets",
        }
        details = "; ".join(
            f"seed={item.get('seed', 'all')} field={item.get('field')} "
            f"({labels.get(str(item.get('field')), item.get('field'))}) "
            f"baseline={item.get('baseline')!r} candidate={item.get('candidate')!r}"
            for item in self.mismatches
        )
        super().__init__(f"matched comparison contract failed: {details}")


@dataclass(frozen=True)
class ComparisonRecord:
    """Normalized evidence needed to establish a matched comparison."""

    seed: int
    provenance: Mapping[str, Any]
    subject_ids: tuple[str, ...]
    true_targets: tuple[float, ...]
    payload: Any


@dataclass(frozen=True)
class MatchedComparison:
    """One exact baseline/candidate pair."""

    seed: int
    baseline: ComparisonRecord
    candidate: ComparisonRecord


def _records_by_seed(
    records: Sequence[ComparisonRecord], *, group_name: str
) -> dict[int, ComparisonRecord]:
    by_seed: dict[int, ComparisonRecord] = {}
    for record in records:
        if isinstance(record.seed, bool) or not isinstance(record.seed, int):
            raise ComparisonContractError(
                (
                    {
                        "field": "seed",
                        "baseline": None if group_name == "candidate" else record.seed,
                        "candidate": record.seed if group_name == "candidate" else None,
                    },
                )
            )
        if record.seed in by_seed:
            raise ComparisonContractError(
                (
                    {
                        "field": f"duplicate_{group_name}_seed",
                        "baseline": record.seed if group_name == "baseline" else None,
                        "candidate": record.seed if group_name == "candidate" else None,
                    },
                )
            )
        by_seed[record.seed] = record
    return by_seed


def match_exact_records(
    baseline_records: Sequence[ComparisonRecord],
    candidate_records: Sequence[ComparisonRecord],
    *,
    provenance_fields: Sequence[str],
) -> tuple[MatchedComparison, ...]:
    """Return seed-sorted pairs only when every declared factor matches."""

    baseline = _records_by_seed(baseline_records, group_name="baseline")
    candidate = _records_by_seed(candidate_records, group_name="candidate")
    baseline_seeds = sorted(baseline)
    candidate_seeds = sorted(candidate)
    if not baseline_seeds or baseline_seeds != candidate_seeds:
        raise ComparisonContractError(
            (
                {
                    "field": "seed_inventory",
                    "baseline": baseline_seeds,
                    "candidate": candidate_seeds,
                },
            )
        )

    mismatches: list[dict[str, Any]] = []
    matched: list[MatchedComparison] = []
    for seed in baseline_seeds:
        baseline_record = baseline[seed]
        candidate_record = candidate[seed]
        for field in provenance_fields:
            baseline_missing = field not in baseline_record.provenance
            candidate_missing = field not in candidate_record.provenance
            baseline_value = baseline_record.provenance.get(field)
            candidate_value = candidate_record.provenance.get(field)
            if (
                baseline_missing
                or candidate_missing
                or baseline_value is None
                or candidate_value is None
                or baseline_value == ""
                or candidate_value == ""
                or baseline_value != candidate_value
            ):
                mismatches.append(
                    {
                        "seed": seed,
                        "field": field,
                        "baseline": baseline_value,
                        "candidate": candidate_value,
                    }
                )
        if baseline_record.subject_ids != candidate_record.subject_ids:
            mismatches.append(
                {
                    "seed": seed,
                    "field": "subject_ids",
                    "baseline": list(baseline_record.subject_ids),
                    "candidate": list(candidate_record.subject_ids),
                }
            )
        if baseline_record.true_targets != candidate_record.true_targets:
            mismatches.append(
                {
                    "seed": seed,
                    "field": "true_targets",
                    "baseline": list(baseline_record.true_targets),
                    "candidate": list(candidate_record.true_targets),
                }
            )
        matched.append(
            MatchedComparison(
                seed=seed,
                baseline=baseline_record,
                candidate=candidate_record,
            )
        )

    if mismatches:
        raise ComparisonContractError(mismatches)
    return tuple(matched)
