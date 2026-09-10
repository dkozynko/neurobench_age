"""Aggregate-only tables and figures for the capacity--data extension."""

from __future__ import annotations

import csv
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from neurobench_age.research.capacity_data_regime_lock import (
    CapacityDataRegimeLockError,
    load_checkpoint_inventory,
    load_final_lock,
    load_prediction_inventory,
)


class CapacityDataRegimeAssetError(ValueError):
    """Raised when aggregate assets are missing, altered, or unsafe."""


_HEAD_LABELS = {
    "mean_linear": "Mean-pooled linear",
    "mean_rich_stats_residual": "Rich-statistics residual",
    "mean_mlp_residual_matched(hidden_dim=4)": "Matched MLP (hidden_dim=4)",
}
_CANDIDATE_HEADS = tuple(name for name in _HEAD_LABELS if name != "mean_linear")
_SIZES = (200, 400, 800)
_FORBIDDEN_PUBLICATION_FIELDS = (
    "subject_id",
    "true_age",
    "prediction",
    "participant",
    "sub-",
    "/" + "workspace/",
    "/" + "Users/",
    "hf_",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapacityDataRegimeAssetError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise CapacityDataRegimeAssetError(f"{description} must be an object")
    return value


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CapacityDataRegimeAssetError(
                f"asset already exists with different content: {path}"
            )
        return
    path.write_bytes(payload)


def _validate_output_root(output_root: Path, repository_root: Path) -> Path:
    output = Path(output_root).resolve()
    forbidden = (
        repository_root / "manuscript/generated",
        repository_root / "results/canonical",
    )
    if any(
        output == root.resolve() or output.is_relative_to(root.resolve())
        for root in forbidden
    ):
        raise CapacityDataRegimeAssetError(
            "extension assets cannot overwrite primary generated assets"
        )
    return output


def _validate_sources(
    *,
    analysis: Mapping[str, Any],
    final_lock_path: Path,
    checkpoint_inventory_path: Path,
    prediction_inventory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        final_lock = load_final_lock(_load_json(final_lock_path, "final extension lock"))
        checkpoint = load_checkpoint_inventory(
            _load_json(checkpoint_inventory_path, "checkpoint inventory"),
            expected_core_digest=final_lock["lock_core_sha256"],
        )
        prediction = load_prediction_inventory(
            _load_json(prediction_inventory_path, "prediction inventory"),
            expected_core_digest=final_lock["lock_core_sha256"],
            expected_checkpoint_digest=final_lock["checkpoint_inventory_body_sha256"],
        )
    except CapacityDataRegimeLockError as error:
        raise CapacityDataRegimeAssetError(str(error)) from error
    if checkpoint["checkpoint_inventory_body_sha256"] != final_lock[
        "checkpoint_inventory_body_sha256"
    ]:
        raise CapacityDataRegimeAssetError("checkpoint inventory is not bound to final lock")
    if prediction["prediction_inventory_body_sha256"] != final_lock[
        "prediction_inventory_body_sha256"
    ]:
        raise CapacityDataRegimeAssetError("prediction inventory is not bound to final lock")
    if analysis.get("final_lock_sha256") != final_lock["lock_sha256"]:
        raise CapacityDataRegimeAssetError("analysis is not bound to final lock")
    if analysis.get("prediction_inventory_body_sha256") != prediction[
        "prediction_inventory_body_sha256"
    ]:
        raise CapacityDataRegimeAssetError(
            "analysis is not bound to prediction inventory"
        )
    training_evidence = analysis.get("training_evidence")
    if not isinstance(training_evidence, Mapping):
        raise CapacityDataRegimeAssetError("analysis training evidence is missing")
    if training_evidence.get("checkpoint_inventory_body_sha256") != checkpoint[
        "checkpoint_inventory_body_sha256"
    ]:
        raise CapacityDataRegimeAssetError(
            "analysis is not bound to checkpoint inventory"
        )
    return final_lock, checkpoint, prediction


def _parameter_counts(checkpoint: Mapping[str, Any]) -> dict[str, int]:
    runs = checkpoint.get("runs")
    if not isinstance(runs, list) or len(runs) != 90:
        raise CapacityDataRegimeAssetError("checkpoint inventory must contain 90 runs")
    by_head: dict[str, set[int]] = {}
    for run in runs:
        head = run.get("head")
        complexity = run.get("head_complexity")
        count = complexity.get("parameter_count") if isinstance(complexity, Mapping) else None
        if not isinstance(head, str) or isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise CapacityDataRegimeAssetError("checkpoint head complexity is invalid")
        by_head.setdefault(head, set()).add(count)
    if set(by_head) != set(_HEAD_LABELS) or any(len(values) != 1 for values in by_head.values()):
        raise CapacityDataRegimeAssetError("head parameter counts are not consistent across runs")
    return {head: next(iter(values)) for head, values in by_head.items()}


def _latex_head(head: str) -> str:
    return _HEAD_LABELS[head].replace("_", r"\_")


def _render_table(rows: Sequence[str], columns: str) -> str:
    return "\n".join(
        [r"\begin{tabular}{" + columns + "}", r"\toprule", *rows, r"\bottomrule", r"\end{tabular}"]
    ) + "\n"


def _fixed_pdf_metadata(title: str) -> dict[str, object]:
    from datetime import datetime, timezone

    timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    return {
        "Title": title,
        "Author": "Anonymous",
        "Subject": "Capacity--data regime aggregate evidence",
        "Keywords": "EEG, REVE, age probing, capacity, data regime",
        "Creator": "neurobench-age",
        "Producer": "Matplotlib",
        "CreationDate": timestamp,
        "ModDate": timestamp,
    }


def _figure_bytes(figure: Any, *, title: str) -> bytes:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="pdf",
        bbox_inches="tight",
        metadata=_fixed_pdf_metadata(title),
    )
    return buffer.getvalue()


def _render_figures(cells: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise CapacityDataRegimeAssetError(
            "Matplotlib is required to render PDF extension figures"
        ) from error

    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.compression": 0,
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    ):
        colors = {
            "mean_rich_stats_residual": "#0072B2",
            "mean_mlp_residual_matched(hidden_dim=4)": "#D55E00",
        }
        absolute, absolute_axis = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
        for head, color in colors.items():
            rows = sorted((cell for cell in cells if cell["head"] == head), key=lambda row: int(row["training_size"]))
            absolute_axis.plot(
                [row["training_size"] for row in rows],
                [row["mean_candidate_pearson"] for row in rows],
                marker="o",
                color=color,
                label=_HEAD_LABELS[head],
            )
        baseline = sorted(
            (cell for cell in cells if cell["head"] == _CANDIDATE_HEADS[0]),
            key=lambda row: int(row["training_size"]),
        )
        absolute_axis.plot(
            [row["training_size"] for row in baseline],
            [row["mean_baseline_pearson"] for row in baseline],
            marker="o",
            color="#333333",
            linestyle="--",
            label=_HEAD_LABELS["mean_linear"],
        )
        absolute_axis.set_xlabel("HBN training subjects")
        absolute_axis.set_ylabel("External Pearson correlation")
        absolute_axis.set_xticks(_SIZES)
        absolute_axis.grid(axis="y", alpha=0.2)
        absolute_axis.legend(frameon=False, fontsize=7)
        absolute_pdf = _figure_bytes(absolute, title="Capacity--data absolute Pearson")
        plt.close(absolute)

        delta, delta_axis = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
        for head, color in colors.items():
            rows = sorted((cell for cell in cells if cell["head"] == head), key=lambda row: int(row["training_size"]))
            means = [float(row["mean_delta"]) for row in rows]
            lows = [means[index] - float(rows[index]["bootstrap"]["ci_low"]) for index in range(3)]
            highs = [float(rows[index]["bootstrap"]["ci_high"]) - means[index] for index in range(3)]
            delta_axis.errorbar(
                _SIZES,
                means,
                yerr=[lows, highs],
                marker="o",
                capsize=3,
                color=color,
                label=_HEAD_LABELS[head],
            )
        delta_axis.axhline(0.0, color="#333333", linestyle="--")
        delta_axis.set_xlabel("HBN training subjects")
        delta_axis.set_ylabel("Candidate − mean-linear Pearson")
        delta_axis.set_xticks(_SIZES)
        delta_axis.grid(axis="y", alpha=0.2)
        delta_axis.legend(frameon=False, fontsize=7)
        delta_pdf = _figure_bytes(delta, title="Capacity--data paired Pearson deltas")
        plt.close(delta)

        seed, seed_axis = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
        for head, color in colors.items():
            for size in _SIZES:
                cell = next(row for row in cells if row["head"] == head and row["training_size"] == size)
                seed_axis.plot(
                    [row["seed"] for row in cell["per_seed"]],
                    [row["pearson_delta"] for row in cell["per_seed"]],
                    marker="o",
                    markersize=3,
                    color=color,
                    alpha=0.55,
                    label=f"{_HEAD_LABELS[head]}, n={size}",
                )
        seed_axis.axhline(0.0, color="#333333", linestyle="--")
        seed_axis.set_xlabel("Training seed")
        seed_axis.set_ylabel("Paired external Pearson delta")
        seed_axis.set_xticks(tuple(range(33, 43)))
        seed_axis.grid(axis="y", alpha=0.2)
        seed_axis.legend(frameon=False, fontsize=6, ncol=2)
        seed_pdf = _figure_bytes(seed, title="Capacity--data seed variability")
        plt.close(seed)

    return {
        "capacity_data_regime_absolute.pdf": absolute_pdf,
        "capacity_data_regime_delta.pdf": delta_pdf,
        "capacity_data_regime_seed_deltas.pdf": seed_pdf,
    }


def _publication_text_is_safe(payload: str, path: Path) -> None:
    lowered = payload.casefold()
    for field in _FORBIDDEN_PUBLICATION_FIELDS:
        if field.casefold() in lowered:
            raise CapacityDataRegimeAssetError(
                f"publication asset contains restricted field {field}: {path}"
            )


def build_capacity_data_regime_assets(
    *,
    analysis_path: Path,
    final_lock_path: Path,
    checkpoint_inventory_path: Path,
    prediction_inventory_path: Path,
    output_root: Path,
    repository_root: Path,
) -> Mapping[str, Any]:
    """Render only aggregate assets bound to the complete finalized evidence chain."""

    analysis = _load_json(analysis_path, "capacity analysis")
    analysis_body = {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    if analysis.get("analysis_sha256") != _canonical_sha256(analysis_body):
        raise CapacityDataRegimeAssetError("analysis digest does not match its content")
    final_lock, checkpoint, prediction = _validate_sources(
        analysis=analysis,
        final_lock_path=final_lock_path,
        checkpoint_inventory_path=checkpoint_inventory_path,
        prediction_inventory_path=prediction_inventory_path,
    )
    cells = analysis.get("cells")
    contrasts = analysis.get("contrasts")
    training_evidence = analysis.get("training_evidence")
    if not isinstance(cells, list) or len(cells) != 6:
        raise CapacityDataRegimeAssetError("analysis must contain all six extension cells")
    if not isinstance(contrasts, list) or len(contrasts) != 6:
        raise CapacityDataRegimeAssetError("analysis must contain all six contrasts")
    if not isinstance(training_evidence, Mapping) or training_evidence.get("run_count") != 90:
        raise CapacityDataRegimeAssetError("analysis must retain the complete 90-run evidence")
    counts = _parameter_counts(checkpoint)
    for cell in cells:
        if cell.get("head") not in counts or cell.get("training_size") not in _SIZES:
            raise CapacityDataRegimeAssetError("analysis cell identity is invalid")
        if len(cell.get("per_seed", [])) != 10:
            raise CapacityDataRegimeAssetError("analysis cell seed inventory is incomplete")

    output = _validate_output_root(output_root, repository_root)
    output.mkdir(parents=True, exist_ok=True)
    import io

    generated: dict[str, bytes] = {}
    cell_csv = io.StringIO(newline="")
    writer = csv.writer(cell_csv)
    writer.writerow(
        [
            "training_size",
            "head",
            "parameter_count",
            "mean_baseline_pearson",
            "mean_candidate_pearson",
            "candidate_pearson_seed_sd",
            "mean_delta",
            "seed_delta_sample_sd",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "seed_randomization_p",
            "holm_adjusted_p",
            "wins",
            "ties",
            "losses",
            "worst_seed_delta",
        ]
    )
    for cell in cells:
        randomization = cell["seed_randomization"]
        writer.writerow(
            [
                cell["training_size"],
                cell["head"],
                counts[cell["head"]],
                cell["mean_baseline_pearson"],
                cell["mean_candidate_pearson"],
                cell["candidate_pearson_seed_sd"],
                cell["mean_delta"],
                cell["seed_delta_sample_sd"],
                cell["bootstrap"]["ci_low"],
                cell["bootstrap"]["ci_high"],
                randomization["p_value"],
                randomization["holm_adjusted_p_value"],
                cell["wins"],
                cell["ties"],
                cell["losses"],
                cell["worst_seed_delta"],
            ]
        )
    generated["capacity_data_regime_cells.csv"] = cell_csv.getvalue().encode()

    cell_rows = [
        "Training size & Head & Params & Mean $\\Delta$ & 95\\% CI & Holm $p$ & W/T/L \\\\",
        r"\midrule",
    ]
    for cell in cells:
        randomization = cell["seed_randomization"]
        cell_rows.append(
            f"{cell['training_size']} & {_latex_head(str(cell['head']))} & {counts[cell['head']]} & {float(cell['mean_delta']):.3f} & [{float(cell['bootstrap']['ci_low']):.3f}, {float(cell['bootstrap']['ci_high']):.3f}] & {float(randomization['holm_adjusted_p_value']):.3f} & {cell['wins']}/{cell['ties']}/{cell['losses']} \\\\"
        )
    generated["capacity_data_regime_cells.tex"] = _render_table(cell_rows, "rlrrrrr").encode()

    contrast_csv = io.StringIO(newline="")
    writer = csv.writer(contrast_csv)
    writer.writerow(["head", "contrast", "observed", "bootstrap_ci_low", "bootstrap_ci_high", "seed_randomization_p", "holm_adjusted_p", "seed_delta_sample_sd"])
    contrast_rows = ["Head & Contrast & Change in $\\Delta$ & 95\\% CI & Holm $p$ & SD \\\\", r"\midrule"]
    for contrast in contrasts:
        randomization = contrast["seed_randomization"]
        writer.writerow(
            [
                contrast["head"],
                contrast["name"],
                contrast["observed"],
                contrast["bootstrap"]["ci_low"],
                contrast["bootstrap"]["ci_high"],
                randomization["p_value"],
                randomization["holm_adjusted_p_value"],
                contrast["seed_delta_sample_sd"],
            ]
        )
        contrast_rows.append(
            f"{_latex_head(str(contrast['head']))} & {contrast['name'].replace('_', r'\_')} & {float(contrast['observed']):.3f} & [{float(contrast['bootstrap']['ci_low']):.3f}, {float(contrast['bootstrap']['ci_high']):.3f}] & {float(randomization['holm_adjusted_p_value']):.3f} & {float(contrast['seed_delta_sample_sd']):.3f} \\\\" 
        )
    generated["capacity_data_regime_contrasts.csv"] = contrast_csv.getvalue().encode()
    generated["capacity_data_regime_contrasts.tex"] = _render_table(contrast_rows, "llrrrr").encode()

    absolute_rows = ["Training size & Head & Params & Mean Pearson & Seed SD \\\\", r"\midrule"]
    for size in _SIZES:
        baseline = next(cell for cell in cells if cell["training_size"] == size)
        absolute_rows.append(
            f"{size} & Mean-linear baseline & {counts['mean_linear']} & {float(baseline['mean_baseline_pearson']):.3f} & {float(baseline['baseline_pearson_seed_sd']):.3f} \\\\"
        )
        for cell in cells:
            if cell["training_size"] != size:
                continue
            absolute_rows.append(
                f"{size} & {_latex_head(str(cell['head']))} & {counts[cell['head']]} & {float(cell['mean_candidate_pearson']):.3f} & {float(cell['candidate_pearson_seed_sd']):.3f} \\\\"
            )
    generated["capacity_data_regime_absolute.tex"] = _render_table(absolute_rows, "llrrr").encode()

    seed_csv = io.StringIO(newline="")
    writer = csv.writer(seed_csv)
    writer.writerow(["training_size", "head", "seed", "pearson_delta"])
    for cell in cells:
        for row in cell["per_seed"]:
            writer.writerow([cell["training_size"], cell["head"], row["seed"], row["pearson_delta"]])
    generated["capacity_data_regime_seed_deltas.csv"] = seed_csv.getvalue().encode()

    validation_lookup: dict[tuple[int, str, int], float] = {}
    for run in training_evidence["runs"]:
        history = run["validation_history"]
        selected = [row for row in history if row.get("epoch") == run["selected_epoch"]]
        if len(selected) != 1 or not isinstance(selected[0].get("validation_subject_pearson"), (int, float)):
            raise CapacityDataRegimeAssetError("training validation history cannot resolve selected Pearson")
        validation_lookup[(run["training_size"], run["head"], run["seed"])] = float(selected[0]["validation_subject_pearson"])
    transfer_rows = ["Training size & Head & HBN validation Pearson & MIPDB external Pearson \\\\", r"\midrule"]
    for cell in cells:
        validation_values = [validation_lookup[(cell["training_size"], cell["head"], row["seed"])] for row in cell["per_seed"]]
        external_values = [float(row["candidate"]["pearson"]) for row in cell["per_seed"]]
        transfer_rows.append(
            f"{cell['training_size']} & {_latex_head(str(cell['head']))} & {sum(validation_values) / len(validation_values):.3f} & {sum(external_values) / len(external_values):.3f} \\\\"
        )
    generated["capacity_data_regime_validation_external_transfer.tex"] = _render_table(transfer_rows, "llrr").encode()
    generated.update(_render_figures(cells))

    for name, payload in generated.items():
        if name.endswith((".csv", ".tex")):
            _publication_text_is_safe(payload.decode("utf-8"), output / name)
        _write_create_only(output / name, payload)

    manifest_body = {
        "schema_version": 1,
        "status": "complete",
        "analysis_sha256": analysis["analysis_sha256"],
        "exploratory_inference_sha256": analysis["exploratory_inference"]["sha256"],
        "final_lock_sha256": final_lock["lock_sha256"],
        "checkpoint_inventory_body_sha256": checkpoint["checkpoint_inventory_body_sha256"],
        "prediction_inventory_body_sha256": prediction["prediction_inventory_body_sha256"],
        "files": {
            name: {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
            for name, payload in sorted(generated.items())
        },
    }
    manifest = {**manifest_body, "asset_manifest_sha256": _canonical_sha256(manifest_body)}
    _write_create_only(
        output / "capacity_data_regime_asset_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest
