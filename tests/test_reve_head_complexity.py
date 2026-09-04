from __future__ import annotations

import pytest

from neurobench_age.heads.math import SUPPORTED_HEAD_VARIANTS, head_complexity_metadata


def test_every_supported_head_has_a_complexity_contract() -> None:
    for variant in SUPPORTED_HEAD_VARIANTS:
        metadata = head_complexity_metadata(variant, embed_dim=512, n_outputs=1)

        assert metadata["variant"] == variant
        assert metadata["input_width"]
        assert metadata["output_width"] == 1
        assert metadata["operations"]
        assert metadata["parameter_count_formula"]


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("mean_linear", 513),
        ("mean_layer_mix", 514),
        ("mean_rich_stats_residual", 2561),
    ],
)
def test_complexity_contract_reports_exact_simple_head_counts(variant: str, expected: int) -> None:
    metadata = head_complexity_metadata(variant, embed_dim=512, n_outputs=1)

    assert metadata["parameter_count"] == expected


def test_complexity_contract_rejects_unknown_variants() -> None:
    with pytest.raises(ValueError, match="unsupported head variant"):
        head_complexity_metadata("not-a-real-head", embed_dim=512, n_outputs=1)
