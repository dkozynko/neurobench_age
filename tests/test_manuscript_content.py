from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
SECTIONS = (
    "abstract.tex",
    "introduction.tex",
    "related_work.tex",
    "datasets.tex",
    "methods.tex",
    "results.tex",
    "discussion.tex",
    "limitations.tex",
    "conclusion.tex",
    "reproducibility.tex",
    "supplementary.tex",
)
GENERATED = (
    "results_macros.tex",
    "main_metrics.tex",
    "confirmatory_comparisons.tex",
    "cohort_summary.tex",
    "seed_deltas.pdf",
    "bootstrap_intervals.pdf",
    "calibration_summary.pdf",
    "assets_manifest.json",
    "assets_manifest.sha256",
)


def _source_text() -> str:
    paths = [MANUSCRIPT / "main.tex", MANUSCRIPT / "macros.tex"] + [
        MANUSCRIPT / "sections" / name for name in SECTIONS
    ]
    paths.append(MANUSCRIPT / "sections" / "capacity_data_regime.tex")
    paths.append(MANUSCRIPT / "sections" / "ds006780_external.tex")
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_manuscript_has_complete_neutral_structure() -> None:
    assert (MANUSCRIPT / "main.tex").is_file()
    assert (MANUSCRIPT / "references.bib").is_file()
    assert (MANUSCRIPT / "Makefile").is_file()
    main = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    assert "When More Expressive Probes Do Not Establish Stable External Gains" in main
    assert r"\author{Anonymous}" in main
    for name in SECTIONS:
        path = MANUSCRIPT / "sections" / name
        assert path.is_file() and len(path.read_text(encoding="utf-8").split()) >= 25
        assert rf"\input{{sections/{name.removesuffix('.tex')}}}" in main
    for name in GENERATED:
        assert (MANUSCRIPT / "generated" / name).is_file()


def test_manuscript_has_no_placeholders_private_paths_or_manual_primary_literals() -> None:
    source = _source_text()
    lowered = source.casefold()
    for forbidden in (
        "todo",
        "tbd",
        "placeholder",
        "/" + "workspace/",
        "/" + "users/",
        "hf_",
    ):
        assert forbidden not in lowered
    for literal in (
        "0.6443718397587508",
        "0.008463260027991882",
        "0.031988284611527654",
        "0.00957098952349893",
        "-0.01886912754348062",
    ):
        assert literal not in source
    assert "does not establish equivalence" in lowered
    assert "mipdb is not an official neuralbench score" in lowered


def test_primary_results_are_loaded_from_generated_assets() -> None:
    source = _source_text()
    results = (MANUSCRIPT / "sections" / "results.tex").read_text(encoding="utf-8")
    main = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    assert r"\input{generated/results_macros}" in main
    for name in ("main_metrics", "confirmatory_comparisons", "cohort_summary"):
        assert rf"\input{{generated/{name}}}" in results or rf"\input{{generated/{name}}}" in source
    for name in ("seed_deltas", "bootstrap_intervals", "calibration_summary"):
        assert rf"generated/{name}" in source
    assert r"\BaselinePearson" in results
    assert r"\LayerLinearPearsonDelta" in results


def test_capacity_extension_is_integrated_into_main_narrative() -> None:
    abstract = (MANUSCRIPT / "sections" / "abstract.tex").read_text(encoding="utf-8")
    introduction = (MANUSCRIPT / "sections" / "introduction.tex").read_text(encoding="utf-8")
    methods = (MANUSCRIPT / "sections" / "methods.tex").read_text(encoding="utf-8")
    results = (MANUSCRIPT / "sections" / "results.tex").read_text(encoding="utf-8")
    discussion = (MANUSCRIPT / "sections" / "discussion.tex").read_text(encoding="utf-8")
    conclusion = (MANUSCRIPT / "sections" / "conclusion.tex").read_text(encoding="utf-8")
    for section in (abstract, introduction, methods, results, discussion, conclusion):
        assert "capacity--data" in section
        assert any(marker in section for marker in ("secondary", "exploratory", "bounded"))
    assert "../results/extensions/capacity_data_regime_v3/capacity_data_regime_delta.pdf" in results
    assert "universal scaling law" in results


def test_layerwise_extension_is_integrated_as_secondary_analysis() -> None:
    abstract = (MANUSCRIPT / "sections" / "abstract.tex").read_text(encoding="utf-8")
    methods = (MANUSCRIPT / "sections" / "methods.tex").read_text(encoding="utf-8")
    results = (MANUSCRIPT / "sections" / "results.tex").read_text(encoding="utf-8")
    discussion = (MANUSCRIPT / "sections" / "discussion.tex").read_text(encoding="utf-8")
    limitations = (MANUSCRIPT / "sections" / "limitations.tex").read_text(encoding="utf-8")
    reproducibility = (MANUSCRIPT / "sections" / "reproducibility.tex").read_text(
        encoding="utf-8"
    )
    main = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")

    for section in (abstract, methods, results, discussion, limitations, reproducibility):
        assert "layer-wise" in section or "layerwise" in section
    assert "exploratory" in results
    assert "final-layer mean-linear" in results
    assert "four" in results and "ten" in results
    assert r"\LayerwiseSubjectCount{}" in results
    assert "stable" in results
    assert "../results/extensions/layerwise_probe_20260910/assets/layerwise_depth.pdf" in results
    assert "../results/extensions/layerwise_probe_20260910/assets/layerwise_seed_deltas.pdf" in results
    assert "../results/extensions/layerwise_probe_20260910/assets/layerwise_summary" in results
    assert r"\input{../results/extensions/layerwise_probe_20260910/assets/layerwise_results_macros}" in main
    assert "crossed zero" in results
    assert "not a primary" in results or "not establish" in results


def test_layerwise_extension_does_not_promote_randomization_p_values() -> None:
    results = (MANUSCRIPT / "sections" / "results.tex").read_text(encoding="utf-8").casefold()
    discussion = (MANUSCRIPT / "sections" / "discussion.tex").read_text(encoding="utf-8").casefold()
    assert "holm" in results
    assert "descriptive" in results or "exploratory" in results
    assert "stable improvement" not in results
    assert "stable improvement" not in discussion


def test_ds006780_extension_is_integrated_as_uncertainty_limited_transfer() -> None:
    section = (MANUSCRIPT / "sections" / "ds006780_external.tex").read_text(
        encoding="utf-8"
    )
    results = (MANUSCRIPT / "sections" / "results.tex").read_text(encoding="utf-8")
    main = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    asset_root = ROOT / "results" / "extensions" / "ds006780_external_v5"

    assert "secondary transfer" in section.casefold()
    assert "precision gate" in section.casefold()
    assert "stable superiority" in section.casefold()
    assert "0.06" not in section  # values are loaded through generated macros
    assert r"\input{sections/ds006780_external}" in results
    assert r"\input{../results/extensions/ds006780_external_v5/ds006780_results_macros}" in main
    for name in (
        "external_analysis.json",
        "precision_gate.json",
        "README.md",
        "assets_manifest.json",
        "ds006780_results_macros.tex",
        "ds006780_summary.tex",
        "ds006780_external.pdf",
    ):
        assert (asset_root / name).is_file()


def test_every_citation_key_exists_and_every_bibliography_entry_is_used() -> None:
    source = _source_text()
    bibliography = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    cited = {
        key.strip()
        for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", source)
        for key in group.split(",")
    }
    declared = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert cited
    assert cited == declared


def test_manuscript_build_inputs_are_portable() -> None:
    source = _source_text()
    assert r"\bibliography{references}" in source
    assert r"\includegraphics" in source
    assert "shell-escape" not in source.casefold()
    private_path_pattern = rf"(?:/{'Users'}/|/{'workspace'}/|(?:^|\s)[A-Za-z]:\\)"
    assert not re.search(private_path_pattern, source)


def test_capacity_data_extension_uses_exact_aggregate_assets() -> None:
    source = (MANUSCRIPT / "sections" / "capacity_data_regime.tex").read_text(
        encoding="utf-8"
    )
    asset_root = ROOT / "results" / "extensions" / "capacity_data_regime_v3"
    required = (
        "capacity_data_regime_absolute.tex",
        "capacity_data_regime_cells.tex",
        "capacity_data_regime_contrasts.tex",
        "capacity_data_regime_validation_external_transfer.tex",
        "capacity_data_regime_absolute.pdf",
        "capacity_data_regime_delta.pdf",
        "capacity_data_regime_seed_deltas.pdf",
    )
    for name in required:
        assert (asset_root / name).is_file()
        assert name in source
    assert "*" not in source
