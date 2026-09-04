from __future__ import annotations

import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
import neurobench_age.pipelines.official as official
import neurobench_age.pipelines.official_runtime as runtime
import neurobench_age.heads.upstream as reve
from neurobench_age.data.selective_download import (
    RELEASE_TO_STUDY_ID,
    SELECTIVE_TASK,
    _audit_release,
    _build_provenance_payload,
    _current_provenance_paths,
)

from neurobench_age.heads.upstream import (
    AdapterContractError,
    ProtocolMismatchError,
    _get_path,
    initialize_mean_anchor_query,
    initialize_last_tuned_query,
    make_upstream_reve_wrapper,
    UPSTREAM_REVE_COMMIT,
    UPSTREAM_REVE_FILE_HASHES,
    UpstreamReveHeadModel,
    source_lock_metadata,
    validate_official_protocol,
    verify_upstream_source_hashes,
)
from neurobench_age.pipelines.official import (
    EpochValidationMetrics,
    _average_state_dicts,
    _ensure_fresh_strict_run_dir,
    _resolve_swa_raw_checkpoint,
    _select_swa_window,
    _set_frozen_experiment_field,
    _run_experiments_synchronously,
    collect_run_artifacts,
    _write_config,
    validate_evaluation_options,
    validate_seeds,
)


def test_strict_run_rejects_reusing_existing_seed_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "mean_linear" / "seed33"
    run_dir.mkdir(parents=True)
    (run_dir / "test_started.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be empty"):
        _ensure_fresh_strict_run_dir(run_dir)


def test_configure_determinism_sets_strict_reproducibility_flags() -> None:
    previous = {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    try:
        official.configure_determinism()
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.use_deterministic_algorithms(previous["algorithms"])
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = previous["cuda_matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]


def test_run_metadata_paths_always_include_canonical_seed_directory(tmp_path: Path) -> None:
    canonical_run_dir = tmp_path / "mean_linear" / "seed34"
    experiment = SimpleNamespace(
        infra=SimpleNamespace(uid_folder=lambda: None),
    )

    paths = runtime._run_metadata_paths(experiment, canonical_run_dir)

    assert paths == [canonical_run_dir / "run_metadata.json"]


def test_swa_window_selection_uses_only_the_declared_late_window() -> None:
    records = [
        {"epoch": 1, "val/pearsonr": 0.40},
        {"epoch": 2, "val/pearsonr": 0.55},
        {"epoch": 3, "val/pearsonr": 0.60},
        {"epoch": 4, "val/pearsonr": 0.50},
        {"epoch": 5, "val/pearsonr": 0.70},
    ]

    selected = _select_swa_window(records, window_size=3)

    assert selected == [
        {"epoch": 3, "val/pearsonr": 0.60},
        {"epoch": 4, "val/pearsonr": 0.50},
        {"epoch": 5, "val/pearsonr": 0.70},
    ]


def test_swa_state_average_preserves_non_floating_buffers() -> None:
    first = {"weight": torch.tensor([1.0, 3.0]), "counter": torch.tensor(2, dtype=torch.long)}
    second = {"weight": torch.tensor([3.0, 5.0]), "counter": torch.tensor(4, dtype=torch.long)}

    averaged = _average_state_dicts([first, second])

    assert torch.equal(averaged["weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(averaged["counter"], second["counter"])


def test_swa_uses_the_official_nested_best_checkpoint(tmp_path: Path) -> None:
    nested = tmp_path / "neuralbench.main.Experiment.run,1" / "seed=33,task_name=age"
    nested.mkdir(parents=True)
    checkpoint = nested / "best.ckpt"
    checkpoint.write_bytes(b"official-checkpoint")
    trainer = SimpleNamespace(
        checkpoint_callback=SimpleNamespace(best_model_path=str(checkpoint))
    )

    assert _resolve_swa_raw_checkpoint(trainer, tmp_path / "seed33") == checkpoint


def test_correlation_auxiliary_loss_keeps_mse_and_returns_named_components() -> None:
    loss = runtime.CorrelationAuxiliaryLoss(torch.nn.MSELoss(), coefficient=0.02)
    prediction = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.0], [2.0], [4.0]])

    values = loss(prediction, target)

    assert torch.isfinite(values)
    assert torch.allclose(loss.last_mse, torch.tensor(1.0 / 3.0))
    assert loss.last_pearson_aux >= 0
    assert values >= loss.last_mse


def test_correlation_auxiliary_loss_is_finite_for_singleton_batches() -> None:
    loss = runtime.CorrelationAuxiliaryLoss(torch.nn.MSELoss(), coefficient=0.02)

    values = loss(torch.tensor([[2.0]]), torch.tensor([[1.0]]))

    assert torch.isfinite(values)
    assert loss.last_pearson_aux == 0


def test_correlation_loss_config_records_the_frozen_coefficient(tmp_path: Path) -> None:
    config_path = tmp_path / "correlation_config.json"

    _write_config(
        config_path,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "results",
        head_variant="mean_rich_stats_residual",
        correlation_loss_lambda=0.02,
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["CORRELATION_LOSS_LAMBDA"] == 0.02
    assert payload["CORRELATION_LOSS_OBJECTIVE"] == "batch_pearson"


def test_layer_mix_config_records_explicit_layer_indices(tmp_path: Path) -> None:
    config_path = tmp_path / "layer_mix_config.json"

    _write_config(
        config_path,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "results",
        head_variant="mean_layer_mix",
        layer_indices=(-2, -1),
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["H7_HEAD_VARIANT"] == "mean_layer_mix"
    assert payload["H7_LAYER_INDICES"] == [-2, -1]


def test_smooth_l1_loss_uses_declared_beta_without_changing_default() -> None:
    mse = runtime.build_training_loss("mse")
    robust = runtime.build_training_loss("smooth_l1")
    assert isinstance(mse, nn.MSELoss)
    assert isinstance(robust, nn.SmoothL1Loss)
    assert robust.beta == 1.0
    with pytest.raises(ValueError, match="robust_loss"):
        runtime.build_training_loss("huber")


def test_target_zscore_scaler_is_train_only_and_inverse_transform_is_original_units() -> None:
    scaler = runtime.TrainingOnlyTargetZScore()
    scaler.partial_fit(torch.tensor([[10.0], [20.0], [30.0]]), split="train")
    before = scaler.statistics_hash()
    with pytest.raises(ValueError, match="training targets"):
        scaler.partial_fit(torch.tensor([[100.0]]), split="validation")
    assert scaler.statistics_hash() == before
    values = scaler.transform(torch.tensor([[20.0], [30.0]]))
    restored = scaler.inverse_transform(values)
    torch.testing.assert_close(restored, torch.tensor([[20.0], [30.0]]))


def test_target_zscore_metadata_records_train_ids_only() -> None:
    metadata = runtime.target_scaler_metadata(
        scaler=runtime.TrainingOnlyTargetZScore(),
        train_subject_ids=["sub-a"],
        train_timeline_ids=["R1/sub-a/task-RestingState/run-1"],
    )
    assert metadata["fit_split"] == "train"
    assert metadata["train_subject_ids"] == ["sub-a"]
    assert metadata["validation_subject_ids"] == []
    assert metadata["test_subject_ids"] == []


def test_selective_config_uses_dedicated_cache_namespace(tmp_path: Path) -> None:
    config_path = tmp_path / "selective_config.json"

    _write_config(
        config_path,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "results",
        data_mode="selective_task",
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["CACHE_DIR"] == str(tmp_path / "data" / "neuralbench_official_cache_selective_task")


def test_selective_eeglab_reader_uses_simplified_mat_structures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mne = ModuleType("mne")
    fake_mne_io = ModuleType("mne.io")
    fake_mne_eeglab = ModuleType("mne.io.eeglab")
    fake_eeglab = ModuleType("mne.io.eeglab.eeglab")
    fake_scipy = ModuleType("scipy")
    fake_scipy_io = ModuleType("scipy.io")
    original_readmat = object()
    calls: dict[str, object] = {}

    def loadmat(*args: object, **kwargs: object) -> object:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {"EEG": "simplified"}

    fake_eeglab._readmat = original_readmat
    fake_scipy_io.loadmat = loadmat
    fake_scipy.io = fake_scipy_io
    fake_mne_eeglab.eeglab = fake_eeglab
    fake_mne_io.eeglab = fake_mne_eeglab
    fake_mne.io = fake_mne_io
    for name, module in {
        "mne": fake_mne,
        "mne.io": fake_mne_io,
        "mne.io.eeglab": fake_mne_eeglab,
        "mne.io.eeglab.eeglab": fake_eeglab,
        "scipy": fake_scipy,
        "scipy.io": fake_scipy_io,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    originals = runtime._install_selective_eeglab_mat_reader()
    try:
        assert fake_eeglab._readmat is not original_readmat
        result = fake_eeglab._readmat(
            "record.set",
            uint16_codec="latin1",
            preload=True,
        )
    finally:
        fake_eeglab._readmat = originals["readmat"]

    assert result == {"EEG": "simplified"}
    assert calls == {
        "args": ("record.set",),
        "kwargs": {
            "struct_as_record": False,
            "squeeze_me": True,
            "simplify_cells": True,
            "uint16_codec": "latin1",
        },
    }
    assert fake_eeglab._readmat is original_readmat


def test_simplified_eeglab_reader_is_enabled_for_manifest_and_selective_modes() -> None:
    assert runtime._should_use_simplified_eeglab_reader("manifest") is True
    assert runtime._should_use_simplified_eeglab_reader("selective_task") is True
    assert runtime._should_use_simplified_eeglab_reader("full") is False


class _FakeReveCore(nn.Module):
    embed_dim = 4
    patch_size = 3
    patch_overlap = 0

    def __init__(self) -> None:
        super().__init__()
        self.seen_pos: torch.Tensor | None = None

    def forward(
        self,
        eeg: torch.Tensor,
        *,
        pos: torch.Tensor | None = None,
        return_output: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
        self.seen_pos = pos
        base = eeg.mean(dim=-1, keepdim=True).expand(-1, eeg.shape[1], self.embed_dim)
        layers = [base, base + 1.0, base + 2.0]
        return layers if return_output else layers[-1]


class _FakeReveWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeReveCore()
        self.register_buffer("channel_indices", None)

    def forward(
        self,
        eeg: torch.Tensor,
        *,
        pos: torch.Tensor | None = None,
) -> torch.Tensor:
        return self.model(eeg, pos=pos, return_output=True)[-1]


class _RngConsumingReveCore(_FakeReveCore):
    def forward(
        self,
        eeg: torch.Tensor,
        *,
        pos: torch.Tensor | None = None,
        return_output: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        torch.rand(1)
        return super().forward(eeg, pos=pos, return_output=return_output)


class _RngConsumingReveWrapper(_FakeReveWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model = _RngConsumingReveCore()


class _TrainDummyFinalTokenEncoder(nn.Module):
    """Small real encoder whose output is a dense final-token sequence."""

    embed_dim = 4
    patch_size = 3
    patch_overlap = 0
    expected_num_tokens = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(p=0.5)
        with torch.no_grad():
            self.projection.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(self.embed_dim, 3) / 10)
        self.seen_eeg: torch.Tensor | None = None
        self.seen_pos: torch.Tensor | None = None
        self.seen_output: torch.Tensor | None = None
        self.seen_training_flags: tuple[bool, ...] | None = None
        self.seen_inference_mode: bool | None = None
        self.seen_autocast: bool | None = None
        self.raise_on_forward = False
        self.emit_special_token = False

    def forward(self, eeg: torch.Tensor, *, pos: torch.Tensor | None = None) -> torch.Tensor:
        self.seen_eeg = eeg
        self.seen_pos = pos
        self.seen_training_flags = tuple(module.training for module in self.modules())
        self.seen_inference_mode = torch.is_inference_mode_enabled()
        self.seen_autocast = torch.is_autocast_enabled(eeg.device.type)
        if self.raise_on_forward:
            raise RuntimeError("synthetic encoder failure")
        self.seen_output = self.projection(eeg)
        if self.emit_special_token:
            self.seen_output = torch.cat((self.seen_output, self.seen_output[:, :1]), dim=1)
        return self.seen_output


class _ForbiddenLoader:
    def __init__(self, split: str) -> None:
        self.split = split
        self.touched = False

    def __iter__(self):
        self.touched = True
        raise AssertionError(f"{self.split} loader must not be touched during build")


class _FakeTrainingBatch:
    def __init__(
        self,
        eeg: torch.Tensor,
        *,
        channel_positions: torch.Tensor | None = None,
        subject_ids: torch.Tensor | None = None,
) -> None:
        self.data = {"eeg": eeg}
        if channel_positions is not None:
            self.data["channel_positions"] = channel_positions
        if subject_ids is not None:
            self.data["subject_ids"] = subject_ids


class _SingleBatchTrainLoader:
    def __init__(self, batch: _FakeTrainingBatch) -> None:
        self.batch = batch
        self.touched = False

    def __iter__(self):
        self.touched = True
        yield self.batch


class _FakeNeuralBenchFactory:
    def __init__(self, train_batch: _FakeTrainingBatch) -> None:
        self.train_loader = _SingleBatchTrainLoader(train_batch)
        self.validation_loader = _ForbiddenLoader("validation")
        self.test_loader = _ForbiddenLoader("test")
        self.seen_train_batch: _FakeTrainingBatch | None = None
        self.seen_dummy_batch: dict[str, torch.Tensor] | None = None

    def build_dummy_batch(self) -> dict[str, torch.Tensor]:
        batch = next(iter(self.train_loader))
        self.seen_train_batch = batch
        self.seen_dummy_batch = {"eeg": batch.data["eeg"][:1]}
        for key in ("channel_positions", "subject_ids"):
            if key in batch.data:
                self.seen_dummy_batch[key] = batch.data[key][:1]
        return self.seen_dummy_batch

    def build_brain_model(
        self,
        wrapper: object,
        model: nn.Module,
        *,
        n_outputs: int,
) -> UpstreamReveHeadModel:
        return wrapper.build(model, self.build_dummy_batch(), n_outputs=n_outputs)


class _SmokeReve(nn.Module):
    """Minimal data-free REVE stand-in for the CPU smoke contract."""

    def __init__(self, *, embed_dim: int, depth: int, n_chans: int, n_times: int, **_: object) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.projection = nn.Linear(n_times, embed_dim)
        self.n_chans = n_chans

    def forward(
        self,
        eeg: torch.Tensor,
        *,
        pos: torch.Tensor | None = None,
        return_output: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        self.seen_pos = pos
        del pos
        base = self.projection(eeg)
        layers = [base + float(layer) for layer in range(self.depth + 1)]
        return layers if return_output else layers[-1]


class _SmokeReveWrapper(nn.Module):
    def __init__(self, model: nn.Module, *, encoder_only: bool) -> None:
        super().__init__()
        assert encoder_only
        self.model = model

    def forward(
        self,
        eeg: torch.Tensor,
        *,
        pos: torch.Tensor | None = None,
) -> torch.Tensor:
        return self.model(eeg, pos=pos, return_output=True)[-1]


def _install_fake_smoke_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    braindecode = ModuleType("braindecode")
    braindecode_models = ModuleType("braindecode.models")
    braindecode_models.REVE = _SmokeReve
    braindecode.models = braindecode_models

    neuraltrain = ModuleType("neuraltrain")
    neuraltrain_models = ModuleType("neuraltrain.models")
    neuraltrain_reve = ModuleType("neuraltrain.models.reve")
    neuraltrain_reve._ReveWrapper = _SmokeReveWrapper
    neuraltrain.models = neuraltrain_models
    neuraltrain_models.reve = neuraltrain_reve

    for name, module in {
        "braindecode": braindecode,
        "braindecode.models": braindecode_models,
        "neuraltrain": neuraltrain,
        "neuraltrain.models": neuraltrain_models,
        "neuraltrain.models.reve": neuraltrain_reve,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _run_fake_main_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    variant: str,
    official_results: list[dict[str, object]] | None = None,
    captured_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("synthetic\n", encoding="utf-8")
    data_root = tmp_path / "data"
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.json"

    monkeypatch.setattr(official, "manifest_sha256", lambda path: "manifest-sha")
    monkeypatch.setattr(official, "load_manifest_timelines", lambda manifest, root: [{"id": "synthetic"}])
    def fake_run_official_subset(**kwargs: object) -> list[dict[str, object]]:
        if captured_kwargs is not None:
            captured_kwargs.update(kwargs)
        return (
            official_results
            if official_results is not None
            else [
                {
                    "validation_pearsonr": 0.42,
                    "test/pearsonr": 0.17,
                }
            ]
)

    monkeypatch.setattr(official, "run_official_subset", fake_run_official_subset)

    result = official.main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--head-variant",
            variant,
            "--evaluation-protocol",
            "legacy",
            "--seeds",
            "33",
        ]
)
    assert result == 0
    report_path = output_dir / variant / "seed33" / "report.json"
    assert report_path.is_file()
    return json.loads(report_path.read_text(encoding="utf-8"))


def _write_selective_acquisition(root: Path, *, releases: tuple[str, ...]) -> None:
    for release in releases:
        download = root / release / "download"
        eeg = download / "sub-NDAR001" / "eeg"
        eeg.mkdir(parents=True)
        (download / "participants.tsv").write_text("participant_id\tage\nsub-NDAR001\t10\n")
        (download / f"{SELECTIVE_TASK}_eeg.json").write_text('{"EEGReference": "Cz"}\n')
        (eeg / f"sub-NDAR001_{SELECTIVE_TASK}_eeg.set").write_bytes(b"set")
        (eeg / f"sub-NDAR001_{SELECTIVE_TASK}_eeg.fdt").write_bytes(b"fdt")
    audits = tuple(_audit_release(root, release) for release in releases)
    payload, raw = _build_provenance_payload(
        data_root=root,
        requested_releases=releases,
        audits=audits,
    )
    provenance_path, digest_path = _current_provenance_paths(root)
    provenance_path.write_bytes(raw)
    digest_path.write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii")


def _write_complete_selective_acquisition(root: Path) -> None:
    _write_selective_acquisition(root, releases=tuple(RELEASE_TO_STUDY_ID))


def test_full_data_source_resolver_accepts_full_mode_without_manifest(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    source = official._resolve_data_source(
        manifest_path=None,
        full_data=True,
        data_root=data_root,
    )

    assert source.data_mode == "full"
    assert source.manifest_path is None
    assert source.data_root == data_root.resolve()


def test_selective_source_resolver_accepts_complete_acquisition_without_manifest(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_complete_selective_acquisition(data_root)

    source = official._resolve_data_source(
        manifest_path=None,
        full_data=False,
        selective_task=True,
        data_root=data_root,
    )

    assert source.data_mode == "selective_task"
    assert source.manifest_path is None
    assert source.data_root == data_root.resolve()
    assert source.acquisition_provenance_path == (
        data_root / "selective_task_provenance.json"
    ).resolve()


def test_selective_source_resolver_rejects_partial_or_tampered_acquisition(
    tmp_path: Path,
) -> None:
    partial_root = tmp_path / "partial"
    partial_root.mkdir()
    _write_selective_acquisition(partial_root, releases=("R1",))
    with pytest.raises(ValueError, match="complete"):
        official._resolve_data_source(
            manifest_path=None,
            full_data=False,
            selective_task=True,
            data_root=partial_root,
        )

    complete_root = tmp_path / "complete"
    complete_root.mkdir()
    _write_complete_selective_acquisition(complete_root)
    provenance_path, digest_path = _current_provenance_paths(complete_root)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload["task"] = "task-contrastChangeDetection"
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    provenance_path.write_bytes(raw)
    digest_path.write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="task"):
        official._resolve_data_source(
            manifest_path=None,
            full_data=False,
            selective_task=True,
            data_root=complete_root,
        )


def test_selective_acquisition_snapshot_is_create_only_per_run(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_complete_selective_acquisition(data_root)
    source = official._resolve_data_source(
        manifest_path=None,
        full_data=False,
        selective_task=True,
        data_root=data_root,
    )
    run_dir = tmp_path / "run"

    first_path, first_digest = official._copy_selective_acquisition_snapshot(source, run_dir)
    second_path, second_digest = official._copy_selective_acquisition_snapshot(source, run_dir)
    assert first_path == second_path
    assert first_digest == second_digest

    first_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="snapshot differs"):
        official._copy_selective_acquisition_snapshot(source, run_dir)


def test_main_accepts_selective_task_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_complete_selective_acquisition(data_root)
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.json"

    def fake_run_official_subset(**kwargs: object) -> list[dict[str, object]]:
        provenance_path = kwargs["provenance_path"]
        assert isinstance(provenance_path, Path)
        provenance_path.write_text(
            json.dumps(
                {
                    "data_mode": "selective_task",
                    "task": SELECTIVE_TASK,
                    "timeline_count": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return [{"validation_pearsonr": 0.42, "test/pearsonr": 0.17}]

    monkeypatch.setattr(official, "run_official_subset", fake_run_official_subset)

    result = official.main(
        [
            "--selective-task",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--head-variant",
            "mean_linear",
            "--evaluation-protocol",
            "legacy",
            "--seeds",
            "33",
        ]
    )

    assert result == 0
    report = json.loads(
        (output_dir / "mean_linear" / "seed33" / "report.json").read_text(encoding="utf-8")
    )
    assert report["data_mode"] == "selective_task"


def test_main_writes_schema_versioned_evidence_for_successful_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_linear")
    run_dir = tmp_path / "results" / "mean_linear" / "seed33"

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    complexity = json.loads((run_dir / "complexity.json").read_text(encoding="utf-8"))

    assert report["status"] == "completed"
    assert manifest["schema_version"] == "1.0"
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 33
    assert manifest["evaluation_mode"] == "validation_only"
    assert manifest["test_access"] == "sealed"
    assert complexity["schema_version"] == "1.0"
    assert complexity["status"] == "completed"
    assert (run_dir / "config.json").is_file()


def test_main_preserves_failure_evidence_when_run_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("synthetic\n", encoding="utf-8")
    data_root = tmp_path / "data"
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.json"

    monkeypatch.setattr(official, "manifest_sha256", lambda path: "manifest-sha")
    monkeypatch.setattr(official, "load_manifest_timelines", lambda manifest, root: [{"id": "synthetic"}])

    def failing_run(**kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(official, "run_official_subset", failing_run)

    result = official.main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--head-variant",
            "mean_linear",
            "--evaluation-protocol",
            "legacy",
            "--seeds",
            "33",
        ]
    )

    assert result == 1
    run_dir = output_dir / "mean_linear" / "seed33"
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert "synthetic training failure" in failure["error"]
    assert manifest["status"] == "failed"
    assert manifest["failure_reason"] == "synthetic training failure"


def test_full_data_source_resolver_rejects_missing_or_ambiguous_sources(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        official._resolve_data_source(
            manifest_path=None,
            full_data=False,
            data_root=data_root,
        )
    with pytest.raises(ValueError, match="exactly one"):
        official._resolve_data_source(
            manifest_path=manifest,
            full_data=True,
            data_root=data_root,
        )
    with pytest.raises(FileNotFoundError):
        official._resolve_data_source(
            manifest_path=None,
            full_data=True,
            data_root=tmp_path / "missing",
        )
    non_directory = tmp_path / "not-a-directory"
    non_directory.write_text("x\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        official._resolve_data_source(
            manifest_path=None,
            full_data=True,
            data_root=non_directory,
        )


def test_full_data_canonicalizes_order_run_and_snapshot_bytes() -> None:
    timelines = [
        {"release": "R2", "subject": "sub-2", "task": "task-RestingState", "run": "run-1"},
        {"release": "R1", "subject": "sub-1", "task": "task-RestingState"},
    ]

    normalized = official._canonical_full_data_timelines(timelines)
    payload, raw = official._full_data_provenance_payload(
        data_root=Path("/data/hbn"),
        timelines=normalized,
    )

    assert normalized == (
        {"release": "R2", "subject": "sub-2", "task": "task-RestingState", "run": "run-1"},
        {"release": "R1", "subject": "sub-1", "task": "task-RestingState", "run": None},
    )
    assert payload["schema_version"] == 1
    assert payload["data_mode"] == "full"
    assert payload["study"] == "Shirazi2024Hbn"
    assert payload["timeline_count"] == 2
    assert payload["timelines"] == list(normalized)
    assert raw.endswith(b"\n")
    assert json.loads(raw.decode("utf-8")) == payload
    assert official._sha256_bytes(raw) == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "timelines, pattern",
    [
        ([{"release": "R1", "subject": "s", "task": "t", "run": 1}], "run"),
        ([{"release": "R1", "subject": "s", "task": 1, "run": None}], "task"),
        ([{"release": "R1", "subject": "s", "task": "t", "extra": "x"}], "keys"),
        ([{"release": "R1", "subject": "s"}], "task"),
        ([{"release": "R1", "subject": "s", "task": "t", "run": None}] * 2, "duplicate"),
        ([], "empty"),
        (["not-a-mapping"], "mapping"),
    ],
)
def test_full_data_canonicalization_rejects_invalid_timelines(
    timelines: object,
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match=pattern):
        official._canonical_full_data_timelines(timelines)


def test_main_full_data_mode_uses_direct_official_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.json"
    captured: dict[str, object] = {}

    def fake_run_official_subset(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        provenance_path = kwargs["provenance_path"]
        assert isinstance(provenance_path, Path)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "data_mode": "full",
                    "study": "Shirazi2024Hbn",
                    "data_root": str(data_root.resolve()),
                    "timelines": [
                        {
                            "release": "R1",
                            "subject": "S001",
                            "task": "task-RestingState",
                            "run": None,
                        }
                    ],
                    "timeline_count": 1,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return [{"test/pearsonr": 0.17}]

    monkeypatch.setattr(official, "run_official_subset", fake_run_official_subset)

    result = official.main(
        [
            "--full-data",
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--evaluation-protocol",
            "legacy",
            "--seeds",
            "33",
        ]
    )

    assert result == 0
    assert captured["data_mode"] == "full"
    assert captured["manifest_path"] is None
    assert captured["data_root"] == data_root.resolve()
    report = json.loads(
        (output_dir / "mean_linear" / "seed33" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["data_mode"] == "full"
    assert report["manifest_path"] is None
    assert report["manifest_sha256"] is None


def test_full_data_strict_selection_binds_provenance_snapshot(
    tmp_path: Path,
) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    provenance = tmp_path / "full_data_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_mode": "full",
                "study": "Shirazi2024Hbn",
                "data_root": str((tmp_path / "data").resolve()),
                "timelines": [
                    {
                        "release": "R1",
                        "subject": "S001",
                        "task": "task-RestingState",
                        "run": None,
                    }
                ],
                "timeline_count": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    record = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=None,
        provenance_path=provenance,
        data_mode="full",
        timeline_count=1,
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_rich_stats_residual",
        strict_final_test=False,
        sha256_file=official._sha256_file,
    )

    assert record["data_mode"] == "full"
    assert record["timeline_count"] == 1
    assert record["manifest_path"] is None
    assert record["manifest_sha256"] is None
    assert record["provenance_path"] == str(provenance.resolve())
    assert len(record["provenance_sha256"]) == 64

    selection_path = tmp_path / "selection.json"
    runtime._publish_json_create_if_absent(selection_path, record)
    report = official._strict_report_fields(
        selection_path=selection_path,
        results=[],
        strict_final_test=False,
    )
    assert report["data_mode"] == "full"
    assert report["timeline_count"] == 1
    assert report["manifest_path"] is None
    assert report["provenance_path"].endswith("full_data_provenance.json")


def test_full_data_patch_keeps_official_timeline_discovery_and_snapshots_after_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_neuralbench = ModuleType("neuralbench")
    fake_data = ModuleType("neuralbench.data")
    fake_main = ModuleType("neuralbench.main")
    fake_cli = ModuleType("neuralbench.cli")
    fake_experiment_config = ModuleType("neuralbench.experiment_config")
    fake_neuralfetch = ModuleType("neuralfetch")
    fake_studies = ModuleType("neuralfetch.studies")
    fake_shirazi = ModuleType("neuralfetch.studies.shirazi2024hbn")

    class FakeStudy:
        _info = SimpleNamespace(num_timelines=999)

        def iter_timelines(self):
            return iter(({"release": "R1", "subject": "S001", "task": "task-RestingState"},))

    class FakeData:
        def __init__(self) -> None:
            self.study = SimpleNamespace(
                _timelines=[
                    {
                        "release": "R1",
                        "subject": "S001",
                        "task": "task-RestingState",
                    }
                ]
            )

        def prepare(self):
            return {"train": object()}

    class FakeExperiment:
        def _test(self):
            return {}

        def setup_run(self):
            return None

        def prepare_pl_module(self):
            return None

        def setup_trainer(self):
            return SimpleNamespace(callbacks=[])

    fake_data.Data = FakeData
    fake_main.Experiment = FakeExperiment
    fake_cli.load_yaml_config = lambda path, *args, **kwargs: {}
    fake_experiment_config.load_yaml_config = lambda path, *args, **kwargs: {}
    fake_shirazi.Shirazi2024Hbn = FakeStudy
    fake_studies.shirazi2024hbn = fake_shirazi
    fake_neuralfetch.studies = fake_studies
    fake_neuralbench.data = fake_data
    fake_neuralbench.main = fake_main
    fake_neuralbench.cli = fake_cli
    fake_neuralbench.experiment_config = fake_experiment_config
    for name, module in {
        "neuralbench": fake_neuralbench,
        "neuralbench.data": fake_data,
        "neuralbench.main": fake_main,
        "neuralbench.cli": fake_cli,
        "neuralbench.experiment_config": fake_experiment_config,
        "neuralfetch": fake_neuralfetch,
        "neuralfetch.studies": fake_studies,
        "neuralfetch.studies.shirazi2024hbn": fake_shirazi,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(
        official,
        "_load_reve_helpers",
        lambda: SimpleNamespace(
            validate_head_variant=lambda variant: None,
            PROTOCOL_CONTRACT={"name": "test"},
        ),
    )
    monkeypatch.setattr(official, "load_manifest_timelines", lambda *args: pytest.fail("manifest discovery used"))

    original_iter = FakeStudy.iter_timelines
    original_info = FakeStudy._info
    provenance_path = tmp_path / "full_data_provenance.json"
    originals = official._patch_official_components(
        None,
        tmp_path,
        tmp_path / "epoch_validation_metrics.jsonl",
        tmp_path / "selection.json",
        data_mode="full",
        provenance_path=provenance_path,
        head_variant="mean_linear",
        seeds=(33,),
    )
    try:
        assert FakeStudy.iter_timelines is original_iter
        assert FakeStudy._info is original_info
        data = FakeData()
        data.prepare()
    finally:
        official._restore_official_components(originals)

    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert payload["data_mode"] == "full"
    assert payload["timeline_count"] == 1
    assert payload["timelines"] == [
        {
            "release": "R1",
            "subject": "S001",
            "task": "task-RestingState",
            "run": None,
        }
    ]


def test_selective_task_patch_keeps_official_discovery_and_snapshots_resting_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_neuralbench = ModuleType("neuralbench")
    fake_data = ModuleType("neuralbench.data")
    fake_main = ModuleType("neuralbench.main")
    fake_cli = ModuleType("neuralbench.cli")
    fake_experiment_config = ModuleType("neuralbench.experiment_config")
    fake_neuralset = ModuleType("neuralset")
    fake_neuralset_events = ModuleType("neuralset.events")
    fake_neuralset_study = ModuleType("neuralset.events.study")
    fake_neuralfetch = ModuleType("neuralfetch")
    fake_studies = ModuleType("neuralfetch.studies")
    fake_shirazi = ModuleType("neuralfetch.studies.shirazi2024hbn")

    class FakeStudyBase:
        def _all_timelines(self):
            if self._timelines is not None:
                return self._timelines
            timelines = list(self.iter_timelines())
            if self._info is not None and self._info.num_timelines != len(timelines):
                raise RuntimeError("Dataset corrupted")
            self._timelines = timelines
            return timelines

    class FakeStudy(FakeStudyBase):
        _info = SimpleNamespace(num_timelines=999)

        def __init__(self) -> None:
            self._timelines = None

        def iter_timelines(self):
            return iter(({"release": "R1", "subject": "S001", "task": "task-RestingState"},))

    class FakeData:
        def __init__(self) -> None:
            self.study = SimpleNamespace(steps={"source": FakeStudy()})

        def prepare(self):
            self.study.steps["source"]._all_timelines()
            self.study = SimpleNamespace(_timelines=None)
            return {"train": object()}

    class FakeExperiment:
        def _test(self):
            return {}

        def setup_run(self):
            return None

        def prepare_pl_module(self):
            return None

        def setup_trainer(self):
            return SimpleNamespace(callbacks=[])

    fake_data.Data = FakeData
    fake_main.Experiment = FakeExperiment
    fake_cli.load_yaml_config = lambda path, *args, **kwargs: {}
    fake_experiment_config.load_yaml_config = lambda path, *args, **kwargs: {}
    fake_neuralset_study.Study = FakeStudyBase
    fake_neuralset_events.study = fake_neuralset_study
    fake_neuralset.events = fake_neuralset_events
    fake_shirazi.Shirazi2024Hbn = FakeStudy
    fake_studies.shirazi2024hbn = fake_shirazi
    fake_neuralfetch.studies = fake_studies
    fake_neuralbench.data = fake_data
    fake_neuralbench.main = fake_main
    fake_neuralbench.cli = fake_cli
    fake_neuralbench.experiment_config = fake_experiment_config
    for name, module in {
        "neuralbench": fake_neuralbench,
        "neuralbench.data": fake_data,
        "neuralbench.main": fake_main,
        "neuralbench.cli": fake_cli,
        "neuralbench.experiment_config": fake_experiment_config,
        "neuralset": fake_neuralset,
        "neuralset.events": fake_neuralset_events,
        "neuralset.events.study": fake_neuralset_study,
        "neuralfetch": fake_neuralfetch,
        "neuralfetch.studies": fake_studies,
        "neuralfetch.studies.shirazi2024hbn": fake_shirazi,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(
        official,
        "_load_reve_helpers",
        lambda: SimpleNamespace(
            validate_head_variant=lambda variant: None,
            PROTOCOL_CONTRACT={"name": "test"},
        ),
    )
    monkeypatch.setattr(runtime, "_install_selective_eeglab_mat_reader", lambda: None)
    acquisition_path = tmp_path / "selective_task_provenance.json"
    acquisition_path.write_text("{}\n", encoding="utf-8")
    acquisition_digest = hashlib.sha256(acquisition_path.read_bytes()).hexdigest()
    acquisition_path.with_suffix(".sha256").write_text(
        acquisition_digest + "\n",
        encoding="ascii",
    )
    provenance_path = tmp_path / "selective_task_timeline_provenance.json"
    original_iter = FakeStudy.iter_timelines
    original_info = FakeStudy._info
    originals = official._patch_official_components(
        None,
        tmp_path,
        tmp_path / "epoch_validation_metrics.jsonl",
        tmp_path / "selection.json",
        data_mode="selective_task",
        provenance_path=provenance_path,
        acquisition_provenance_path=acquisition_path,
        acquisition_provenance_sha256=acquisition_digest,
        head_variant="mean_linear",
        seeds=(33,),
    )
    try:
        assert FakeStudy.iter_timelines is original_iter
        assert FakeStudy._info is original_info
        data = FakeData()
        data.prepare()
    finally:
        official._restore_official_components(originals)

    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert payload["data_mode"] == "selective_task"
    assert payload["task"] == "task-RestingState"
    assert payload["timeline_count"] == 1


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().to(device="cpu").contiguous().numpy().tobytes()).hexdigest()


def test_tensor_sha256_supports_bfloat16_values() -> None:
    tensor = torch.tensor([1.0, -2.5], dtype=torch.bfloat16)
    expected = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    assert reve._tensor_sha256(tensor) == expected


def _install_fake_neuralbench(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDownstreamWrapper:
        def __init__(self, **kwargs: object) -> None:
            self.on_the_fly_preprocessor = None
            self.channel_adapter_config = None
            self.model_output_key = None
            self.layers_to_freeze = None
            self.layers_to_unfreeze = None
            for name, value in kwargs.items():
                setattr(self, name, value)

    neuralbench = ModuleType("neuralbench")
    modules = ModuleType("neuralbench.modules")
    modules.DownstreamWrapper = FakeDownstreamWrapper
    neuralbench.modules = modules
    monkeypatch.setitem(sys.modules, "neuralbench", neuralbench)
    monkeypatch.setitem(sys.modules, "neuralbench.modules", modules)


def _protocol_config() -> SimpleNamespace:
    return SimpleNamespace(
        trainer_config=SimpleNamespace(n_epochs=40, monitor="val/pearsonr", mode="max", patience=7, gradient_clip_val=1.0, precision="32-true"),
        lightning_optimizer_config=SimpleNamespace(
            optimizer=SimpleNamespace(name="AdamW", lr=1e-4, kwargs={"weight_decay": 0.05}),
            scheduler=SimpleNamespace(name="OneCycleLR", kwargs={"max_lr": 1e-4, "pct_start": 0.1, "anneal_strategy": "cos",}),
),
        loss=SimpleNamespace(name="MSELoss"),
        data=SimpleNamespace(
            batch_size=64,
            num_workers=2,
            persistent_workers=True,
            seed=33,
            duration=2.0,
            stride=2.0,
            neuro=SimpleNamespace(frequency=200.0, filter=[0.5, 99.5], notch_filter=None, scaler="StandardScaler", clamp=15),
),
        target_scaler=None,
)


def _last_tuned_experiment_config() -> SimpleNamespace:
    config = _protocol_config()
    config.checkpoint_selection = SimpleNamespace(monitor="val/pearsonr", mode="max", test_pearsonr_role="diagnostic_only")
    return config


def _last_tuned_optimizer_config() -> dict[str, object]:
    return {
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "kwargs": {"weight_decay": 0.05},
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {
                "max_lr": [1e-4, 1e-5],
                "pct_start": 0.1,
                "anneal_strategy": "cos",
                "div_factor": 25.0,
                "final_div_factor": 1e4,
            },
            "interval": "step",
            "frequency": 1,
        },
        "param_groups": [
            {
                "name": "base",
                "parameter_names": [
                    "encoder.backbone.weight",
                    "head.rms_norm.weight",
                    "head.linear.weight",
                    "head.linear.bias",
                    "head.gate_logit",
                ],
                "learning_rate": 1e-4,
                "weight_decay": 0.05,
            },
            {
                "name": "query",
                "parameter_names": ["head.query_token"],
                "learning_rate": 1e-5,
                "weight_decay": 0.05,
            },
        ],
    }


def test_last_tuned_validator_accepts_full_fixed_protocol_and_preserves_one_arg_api() -> None:
    experiment = _last_tuned_experiment_config()
    optimizer_config = _last_tuned_optimizer_config()

    assert reve.validate_last_tuned_protocol("last_tuned") == "last_tuned"
    assert (
        reve.validate_last_tuned_protocol("last_tuned", experiment=experiment, optimizer_config=optimizer_config)
        == "last_tuned"
)
    assert experiment.trainer_config.monitor == "val/pearsonr"
    assert experiment.checkpoint_selection.monitor == "val/pearsonr"
    assert experiment.checkpoint_selection.test_pearsonr_role == "diagnostic_only"
    assert optimizer_config["optimizer"]["name"] == "AdamW"
    assert optimizer_config["scheduler"]["name"] == "OneCycleLR"
    assert optimizer_config["scheduler"]["kwargs"]["max_lr"] == [1e-4, 1e-5]
    assert optimizer_config["param_groups"][1]["parameter_names"] == [
        "head.query_token"
    ]


@pytest.mark.parametrize("variant", ("mean_linear", "last_avg", "last", "all"))
def test_last_tuned_validator_rejects_invalid_variants_with_extended_api(
    variant: str,
) -> None:
    with pytest.raises((ProtocolMismatchError, ValueError)):
        reve.validate_last_tuned_protocol(variant, experiment=_last_tuned_experiment_config(), optimizer_config=_last_tuned_optimizer_config())


@pytest.mark.parametrize(
    "mutation",
    (
        "trainer_n_epochs",
        "trainer_monitor",
        "trainer_mode",
        "trainer_patience",
        "trainer_gradient_clip_val",
        "trainer_precision",
        "optimizer_name",
        "optimizer_lr",
        "optimizer_weight_decay",
        "scheduler_name",
        "scheduler_max_lr",
        "scheduler_pct_start",
        "scheduler_anneal_strategy",
        "scheduler_div_factor",
        "scheduler_final_div_factor",
        "scheduler_interval",
        "scheduler_frequency",
        "loss_name",
        "data_batch_size",
        "data_num_workers",
        "data_persistent_workers",
        "data_seed",
        "data_duration",
        "data_stride",
        "data_neuro_frequency",
        "data_neuro_filter",
        "data_neuro_notch_filter",
        "data_neuro_scaler",
        "data_neuro_clamp",
        "target_scaler",
        "checkpoint_monitor",
        "checkpoint_mode",
        "checkpoint_test_pearsonr_role",
        "base_group_name",
        "base_group_membership",
        "base_group_learning_rate",
        "base_group_weight_decay",
        "query_group_name",
        "query_group_membership",
        "query_group_learning_rate",
        "query_group_weight_decay",
),
)
def test_last_tuned_validator_rejects_any_fixed_contract_mismatch(
    mutation: str,
) -> None:
    experiment = _last_tuned_experiment_config()
    optimizer_config = _last_tuned_optimizer_config()

    if mutation == "trainer_n_epochs":
        experiment.trainer_config.n_epochs = 41
    elif mutation == "trainer_monitor":
        experiment.trainer_config.monitor = "test/pearsonr"
    elif mutation == "trainer_mode":
        experiment.trainer_config.mode = "min"
    elif mutation == "trainer_patience":
        experiment.trainer_config.patience = 8
    elif mutation == "trainer_gradient_clip_val":
        experiment.trainer_config.gradient_clip_val = 0.5
    elif mutation == "trainer_precision":
        experiment.trainer_config.precision = "16-mixed"
    elif mutation == "optimizer_name":
        optimizer_config["optimizer"]["name"] = "SGD"
    elif mutation == "optimizer_lr":
        optimizer_config["optimizer"]["lr"] = 2e-4
    elif mutation == "optimizer_weight_decay":
        optimizer_config["optimizer"]["kwargs"]["weight_decay"] = 0.01
    elif mutation == "scheduler_name":
        optimizer_config["scheduler"]["name"] = "CosineAnnealingLR"
    elif mutation == "scheduler_max_lr":
        optimizer_config["scheduler"]["kwargs"]["max_lr"] = [1e-4, 2e-5]
    elif mutation == "scheduler_pct_start":
        optimizer_config["scheduler"]["kwargs"]["pct_start"] = 0.2
    elif mutation == "scheduler_anneal_strategy":
        optimizer_config["scheduler"]["kwargs"]["anneal_strategy"] = "linear"
    elif mutation == "scheduler_div_factor":
        optimizer_config["scheduler"]["kwargs"]["div_factor"] = 20.0
    elif mutation == "scheduler_final_div_factor":
        optimizer_config["scheduler"]["kwargs"]["final_div_factor"] = 1e3
    elif mutation == "scheduler_interval":
        optimizer_config["scheduler"]["interval"] = "epoch"
    elif mutation == "scheduler_frequency":
        optimizer_config["scheduler"]["frequency"] = 2
    elif mutation == "loss_name":
        experiment.loss.name = "L1Loss"
    elif mutation == "data_batch_size":
        experiment.data.batch_size = 32
    elif mutation == "data_num_workers":
        experiment.data.num_workers = 0
    elif mutation == "data_persistent_workers":
        experiment.data.persistent_workers = False
    elif mutation == "data_seed":
        experiment.data.seed = 34
    elif mutation == "data_duration":
        experiment.data.duration = 1.0
    elif mutation == "data_stride":
        experiment.data.stride = 1.0
    elif mutation == "data_neuro_frequency":
        experiment.data.neuro.frequency = 250.0
    elif mutation == "data_neuro_filter":
        experiment.data.neuro.filter = [1.0, 45.0]
    elif mutation == "data_neuro_notch_filter":
        experiment.data.neuro.notch_filter = 50.0
    elif mutation == "data_neuro_scaler":
        experiment.data.neuro.scaler = "RobustScaler"
    elif mutation == "data_neuro_clamp":
        experiment.data.neuro.clamp = 10
    elif mutation == "target_scaler":
        experiment.target_scaler = "StandardScaler"
    elif mutation == "checkpoint_monitor":
        experiment.checkpoint_selection.monitor = "test/pearsonr"
    elif mutation == "checkpoint_mode":
        experiment.checkpoint_selection.mode = "min"
    elif mutation == "checkpoint_test_pearsonr_role":
        experiment.checkpoint_selection.test_pearsonr_role = "used_for_selection"
    elif mutation == "base_group_name":
        optimizer_config["param_groups"][0]["name"] = "backbone"
    elif mutation == "base_group_membership":
        optimizer_config["param_groups"][0]["parameter_names"] = [
            "encoder.backbone.weight",
            "head.rms_norm.weight",
            "head.linear.weight",
            "head.linear.bias",
        ]
    elif mutation == "base_group_learning_rate":
        optimizer_config["param_groups"][0]["learning_rate"] = 2e-4
    elif mutation == "base_group_weight_decay":
        optimizer_config["param_groups"][0]["weight_decay"] = 0.01
    elif mutation == "query_group_name":
        optimizer_config["param_groups"][1]["name"] = "query_only"
    elif mutation == "query_group_membership":
        optimizer_config["param_groups"][1]["parameter_names"] = ["head.gate_logit"]
    elif mutation == "query_group_learning_rate":
        optimizer_config["param_groups"][1]["learning_rate"] = 2e-5
    elif mutation == "query_group_weight_decay":
        optimizer_config["param_groups"][1]["weight_decay"] = 0.01
    else:  # pragma: no cover - guarded by parametrize
        raise AssertionError(mutation)

    with pytest.raises((ProtocolMismatchError, ValueError)):
        reve.validate_last_tuned_protocol("last_tuned", experiment=experiment, optimizer_config=optimizer_config)


@pytest.mark.parametrize("variant", ("last_avg", "last", "mean_linear_detached", "mean_linear_warmup", "mean_linear_gradient_scaled", "mean_anchor", "mean_residual", "mean_vector_anchor", "mean_mlp_residual", "mean_stats_residual", "mean_stats_residual_detached", "mean_stats_residual_gradient_scaled"))
def test_official_adapter_ignores_channel_positions_for_final_token_heads(variant: str) -> None:
    encoder = _FakeReveWrapper()
    model = UpstreamReveHeadModel(encoder, variant=variant, n_outputs=1, dropout=0.0)
    eeg = torch.randn(2, 3, 5)
    positions = torch.randn(2, 3, 3)

    output = model(eeg, channel_positions=positions)

    assert output.shape == (2, 1)
    assert encoder.model.seen_pos is None


def test_official_adapter_ignores_channel_positions_for_last_tuned() -> None:
    encoder = _FakeReveWrapper()
    model = UpstreamReveHeadModel(
        encoder,
        variant="last_tuned",
        n_outputs=1,
        dropout=0.0,
        query_token=torch.ones(1, 1, encoder.model.embed_dim),
    )
    eeg = torch.randn(2, 3, 5)
    positions = torch.randn(2, 3, 3)

    output = model(eeg, channel_positions=positions)

    assert output.shape == (2, 1)
    assert encoder.model.seen_pos is None


def test_official_adapter_all_uses_initial_sequence_and_each_layer() -> None:
    encoder = _FakeReveWrapper()
    model = UpstreamReveHeadModel(encoder, variant="all", n_outputs=1, dropout=0.0)
    eeg = torch.randn(2, 3, 5)
    positions = torch.randn(2, 3, 3)

    output = model(eeg, channel_positions=positions)

    assert output.shape == (2, 1)
    assert encoder.model.seen_pos is None


@pytest.mark.parametrize("variant", ("mean_layer_linear", "mean_layer_mix", "mean_layer_mix_fixed"))
def test_official_adapter_layer_heads_use_ordered_outputs_without_positions(variant: str) -> None:
    encoder = _FakeReveWrapper()
    model = UpstreamReveHeadModel(
        encoder,
        variant=variant,
        n_outputs=1,
        dropout=0.0,
        layer_index=-1,
    )
    eeg = torch.randn(2, 3, 5)
    positions = torch.randn(2, 3, 3)

    output = model(eeg, channel_positions=positions)

    assert output.shape == (2, 1)
    assert encoder.model.seen_pos is None
    assert model.head.metadata()["positional_input_excluded"] is True


def test_layer_mix_cli_smoke_reports_local_screening_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_layer_mix"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_layer_mix"
    assert result["prediction_shape"] == [2, 1]


def test_layer_mix_cli_smoke_preserves_explicit_layer_indices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(
        ["--smoke-head", "mean_layer_mix", "--layer-indices", "-2", "-1"]
    ) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_layer_mix"
    assert result["layer_indices_requested"] == [-2, -1]


def test_fixed_layer_mix_cli_smoke_records_frozen_alpha(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(
        [
            "--smoke-head",
            "mean_layer_mix_fixed",
            "--layer-indices",
            "-2",
            "-1",
            "--layer-mix-alpha",
            "0.5",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_layer_mix_fixed"
    assert result["layer_indices_requested"] == [-2, -1]
    assert result["head_metadata"]["fixed_alpha"] == pytest.approx(0.5)
    assert result["head_metadata"]["alpha_trainable"] is False


@pytest.mark.parametrize("device_type", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
def test_last_tuned_initializer_uses_exact_train_dummy_with_safe_runtime_state(
    device_type: str,
) -> None:
    device = torch.device(device_type)
    encoder = _TrainDummyFinalTokenEncoder().to(device)
    encoder.train()
    encoder.projection.eval()
    encoder.dropout.train()
    original_training_flags = tuple(module.training for module in encoder.modules())
    training_batch = _FakeTrainingBatch(
        torch.arange(18, dtype=torch.float32, device=device).reshape(3, 2, 3),
        channel_positions=torch.randn(3, 2, 3, device=device),
        subject_ids=torch.tensor([101, 102, 103], device=device),
)
    factory = _FakeNeuralBenchFactory(training_batch)
    dummy_batch = factory.build_dummy_batch()
    torch.testing.assert_close(dummy_batch["eeg"], training_batch.data["eeg"][:1])
    assert (
        dummy_batch["eeg"].untyped_storage().data_ptr()
        == training_batch.data["eeg"].untyped_storage().data_ptr()
)
    cpu_rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
        query, metadata = initialize_last_tuned_query(encoder, dummy_batch, provenance=reve._NEURALBENCH_TRAIN_DUMMY_CONTEXT)
        assert torch.is_autocast_enabled(device.type)

    expected = encoder.projection(dummy_batch["eeg"]).mean(dim=1, keepdim=True)

    assert encoder.seen_eeg is dummy_batch["eeg"]
    assert encoder.seen_pos is None
    assert encoder.seen_training_flags == (False,) * len(original_training_flags)
    assert encoder.seen_inference_mode is True
    assert encoder.seen_autocast is False
    assert tuple(query.shape) == (1, 1, encoder.embed_dim)
    assert query.dtype == dummy_batch["eeg"].dtype
    assert query.device == dummy_batch["eeg"].device
    assert not query.is_inference()
    assert encoder.seen_output is not None
    assert encoder.seen_output.shape[1] == encoder.expected_num_tokens
    assert query.untyped_storage().data_ptr() != encoder.seen_output.untyped_storage().data_ptr()
    assert query.untyped_storage().data_ptr() != dummy_batch["eeg"].untyped_storage().data_ptr()
    assert torch.isfinite(query).all()
    torch.testing.assert_close(query, expected)
    assert metadata["query_initialization"] == "train_dummy_final_token_mean"
    assert metadata["query_initialization_batch_element"] == 0
    assert metadata["query_initialization_input_shape"] == list(dummy_batch["eeg"].shape)
    assert metadata["query_initialization_input_dtype"] == str(dummy_batch["eeg"].dtype)
    assert metadata["query_initialization_input_device"] == str(dummy_batch["eeg"].device)
    assert metadata["query_initialization_input_sha256"] == _tensor_sha256(dummy_batch["eeg"])
    assert metadata["query_initialization_query_sha256"] == _tensor_sha256(query)
    assert metadata["query_initialization_subject_ids"] == [101]
    assert tuple(module.training for module in encoder.modules()) == original_training_flags
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert all(torch.equal(after, before) for after, before in zip(torch.cuda.get_rng_state_all(), cuda_rng_before))


def test_mean_anchor_initializer_uses_the_first_train_dummy_sample() -> None:
    encoder = _TrainDummyFinalTokenEncoder()
    training_batch = _FakeTrainingBatch(
        torch.arange(18, dtype=torch.float32).reshape(3, 2, 3),
        channel_positions=torch.randn(3, 2, 3),
    )
    dummy_batch = _FakeNeuralBenchFactory(training_batch).build_dummy_batch()

    query, metadata = initialize_mean_anchor_query(
        encoder,
        dummy_batch,
        provenance=reve._NEURALBENCH_TRAIN_DUMMY_CONTEXT,
    )

    expected = encoder.projection(dummy_batch["eeg"]).mean(dim=1, keepdim=True)
    torch.testing.assert_close(query, expected)
    assert metadata["query_initialization"] == "train_dummy_final_token_mean"
    assert encoder.seen_pos is None


@pytest.mark.parametrize("provenance", (None, "neuralbench_train_dummy", object(), torch.ones(1, 2, 3)))
def test_last_tuned_initializer_rejects_missing_or_forgeable_provenance_context(
    provenance: object,
) -> None:
    training_batch = _FakeTrainingBatch(torch.ones(3, 2, 3))
    with pytest.raises(AdapterContractError, match="provenance|context"):
        initialize_last_tuned_query(
            _TrainDummyFinalTokenEncoder(),
            _FakeNeuralBenchFactory(training_batch).build_dummy_batch(),
            provenance=provenance,
)


def test_last_tuned_initializer_rejects_special_or_padded_token_output() -> None:
    encoder = _TrainDummyFinalTokenEncoder()
    encoder.emit_special_token = True
    dummy_batch = _FakeNeuralBenchFactory(_FakeTrainingBatch(torch.ones(3, 2, 3))).build_dummy_batch()

    with pytest.raises(AdapterContractError, match="token count"):
        initialize_last_tuned_query(encoder, dummy_batch, provenance=reve._NEURALBENCH_TRAIN_DUMMY_CONTEXT)


def test_last_tuned_initializer_restores_runtime_state_after_encoder_failure() -> None:
    encoder = _TrainDummyFinalTokenEncoder()
    encoder.train()
    encoder.projection.eval()
    encoder.dropout.train()
    encoder.raise_on_forward = True
    original_training_flags = tuple(module.training for module in encoder.modules())
    cpu_rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training_batch = _FakeTrainingBatch(torch.ones(3, 2, 3))

    with pytest.raises(RuntimeError, match="synthetic encoder failure"):
        initialize_last_tuned_query(
            encoder,
            _FakeNeuralBenchFactory(training_batch).build_dummy_batch(),
            provenance=reve._NEURALBENCH_TRAIN_DUMMY_CONTEXT,
)

    assert encoder.seen_inference_mode is True
    assert encoder.seen_autocast is False
    assert tuple(module.training for module in encoder.modules()) == original_training_flags
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert all(torch.equal(after, before) for after, before in zip(torch.cuda.get_rng_state_all(), cuda_rng_before))


def test_last_tuned_wrapper_build_uses_train_dummy_helper_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    channel_positions = torch.randn(3, 3, 3)
    training_batch = _FakeTrainingBatch(torch.randn(3, 3, 5), channel_positions=channel_positions)
    factory = _FakeNeuralBenchFactory(training_batch)
    expected_train_dummy = training_batch.data["eeg"][:1]
    calls: list[tuple[nn.Module, object, object]] = []
    initialized_query = torch.full((1, 1, encoder.model.embed_dim), 0.25)
    initialization_metadata = {
        "query_initialization": "train_dummy_final_token_mean",
        "query_initialization_batch_element": 0,
        "query_initialization_input_shape": list(expected_train_dummy.shape),
        "query_initialization_input_dtype": str(expected_train_dummy.dtype),
        "query_initialization_input_device": str(expected_train_dummy.device),
        "query_initialization_input_sha256": "train-dummy-sha256",
        "query_initialization_query_sha256": "initialized-query-sha256",
    }

    def fake_initialize(
        received_encoder: nn.Module,
        received_dummy_batch: object,
        *,
        provenance: object,
) -> tuple[torch.Tensor, dict[str, object]]:
        calls.append((received_encoder, received_dummy_batch, provenance))
        return initialized_query, initialization_metadata

    monkeypatch.setattr("neurobench_age.heads.upstream.initialize_last_tuned_query", fake_initialize)

    head_model = factory.build_brain_model(make_upstream_reve_wrapper(variant="last_tuned"), encoder, n_outputs=1)
    assert factory.seen_dummy_batch is not None
    dummy_batch = factory.seen_dummy_batch
    train_dummy = dummy_batch["eeg"]
    positions = dummy_batch["channel_positions"]
    assert encoder.model.seen_pos is None
    output = head_model(train_dummy, channel_positions=positions)

    assert calls == [(encoder, dummy_batch, reve._NEURALBENCH_TRAIN_DUMMY_CONTEXT)]
    received_encoder, received_dummy_batch, received_context = calls[0]
    assert received_encoder is encoder
    assert received_dummy_batch is dummy_batch
    assert received_dummy_batch["eeg"] is train_dummy
    assert received_context is reve._NEURALBENCH_TRAIN_DUMMY_CONTEXT
    assert factory.train_loader.touched
    assert factory.seen_train_batch is training_batch
    assert factory.seen_dummy_batch is dummy_batch
    assert dummy_batch["eeg"] is factory.seen_dummy_batch["eeg"]
    torch.testing.assert_close(dummy_batch["eeg"], training_batch.data["eeg"][:1])
    assert (
        dummy_batch["eeg"].untyped_storage().data_ptr()
        == training_batch.data["eeg"].untyped_storage().data_ptr()
)
    assert not factory.validation_loader.touched
    assert not factory.test_loader.touched
    assert encoder.model.seen_pos is None
    assert output.shape == (1, 1)
    assert head_model.head.query_initialization == "train_dummy_final_token_mean"
    assert (
        head_model.head.tuning_metadata["query_initialization_input_sha256"]
        == "train-dummy-sha256"
)
    assert (
        head_model.head.tuning_metadata["query_initialization_query_sha256"]
        == "initialized-query-sha256"
)


@pytest.mark.parametrize("variant", ("last", "all"))
def test_wrapper_build_preserves_official_last_and_all_behavior(
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("official variants must not initialize from train dummy")

    monkeypatch.setattr("neurobench_age.heads.upstream.initialize_last_tuned_query", fail_if_called)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(1, 3, 5)
    positions = torch.randn(1, 3, 3)

    head_model = make_upstream_reve_wrapper(variant=variant).build(encoder, {"eeg": eeg}, n_outputs=1)
    output = head_model(eeg, channel_positions=positions)

    assert output.shape == (1, 1)
    assert encoder.model.seen_pos is None
    if variant == "last":
        assert head_model.head.query_initialization == "upstream_random"
    else:
        assert head_model.all_layer_encoder is not None
        layers = head_model.all_layer_encoder(eeg, pos=positions)
        assert len(layers) == 3
        assert encoder.model.seen_pos is None
        torch.testing.assert_close(layers[1], layers[0] + 1.0)
        torch.testing.assert_close(layers[2], layers[0] + 2.0)


def test_wrapper_build_mean_linear_copy_uses_only_mean_and_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)
    positions = torch.randn(2, 3, 3)

    head_model = make_upstream_reve_wrapper(variant="mean_linear_copy").build(
        encoder, {"eeg": eeg}, n_outputs=1
    )
    output = head_model(eeg, channel_positions=positions)
    embedding = head_model(eeg, channel_positions=positions, return_embedding=True)

    assert output.shape == (2, 1)
    assert embedding.shape == (2, 4)
    assert encoder.model.seen_pos is None
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
    }
    assert not hasattr(head_model.head, "query_token")


def test_wrapper_build_mean_linear_detached_does_not_update_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5, requires_grad=True)

    head_model = make_upstream_reve_wrapper(variant="mean_linear_detached").build(
        encoder, {"eeg": eeg[:1]}, n_outputs=1
    )
    head_model(eeg).sum().backward()

    assert eeg.grad is None
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
    }


def test_wrapper_build_mean_anchor_starts_as_mean_linear_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)
    positions = torch.randn(2, 3, 3)

    head_model = make_upstream_reve_wrapper(variant="mean_anchor").build(
        encoder, {"eeg": eeg[:1]}, n_outputs=1
    )
    output = head_model(eeg, channel_positions=positions)
    embedding = head_model(eeg, channel_positions=positions, return_embedding=True)

    assert output.shape == (2, 1)
    assert embedding.shape == (2, 4)
    assert encoder.model.seen_pos is None
    assert set(dict(head_model.head.named_parameters())) == {
        "query_token",
        "gamma",
        "linear.weight",
        "linear.bias",
    }
    assert head_model.head.query_initialization == "train_dummy_final_token_mean"
    assert torch.isfinite(head_model.head.query_token).all()
    assert head_model.head.query_token.abs().sum().item() > 0.0
    torch.testing.assert_close(embedding, encoder(eeg).mean(dim=1))


def test_wrapper_build_mean_residual_starts_as_mean_linear_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(variant="mean_residual").build(
        encoder, {"eeg": eeg[:1]}, n_outputs=1
    )
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert head_model.head.query_initialization == "train_dummy_final_token_mean"
    assert torch.isfinite(head_model.head.query_token).all()
    assert head_model.head.correction.weight.abs().sum().item() == 0.0
    assert set(dict(head_model.head.named_parameters())) == {
        "query_token",
        "linear.weight",
        "linear.bias",
        "correction.weight",
    }


def test_wrapper_build_mean_vector_anchor_starts_as_mean_linear_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(variant="mean_vector_anchor").build(
        encoder, {"eeg": eeg[:1]}, n_outputs=1
    )
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert head_model.head.query_initialization == "train_dummy_final_token_mean"
    assert torch.isfinite(head_model.head.query_token).all()
    assert torch.equal(head_model.head.gamma, torch.zeros_like(head_model.head.gamma))
    assert set(dict(head_model.head.named_parameters())) == {
        "query_token",
        "gamma",
        "linear.weight",
        "linear.bias",
    }


def test_wrapper_build_mean_mlp_residual_starts_as_mean_linear_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(variant="mean_mlp_residual").build(
        encoder, {"eeg": eeg[:1]}, n_outputs=1
    )
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert torch.equal(
        head_model.head.correction.weight,
        torch.zeros_like(head_model.head.correction.weight),
    )
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "hidden.weight",
        "hidden.bias",
        "correction.weight",
    }


def test_wrapper_build_mean_stats_residual_starts_as_mean_linear_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(variant="mean_stats_residual").build(
        encoder, {"eeg": eeg[:1]}, n_outputs=1
    )
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert torch.equal(
        head_model.head.correction.weight,
        torch.zeros_like(head_model.head.correction.weight),
    )
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "correction.weight",
    }


def test_wrapper_build_mean_anchor_ensemble_starts_at_mean_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(variant="mean_anchor_ensemble").build(
        encoder, {"eeg": eeg[:1]}, n_outputs=1
    )
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert head_model.head.gate_value == pytest.approx(0.0)
    assert head_model.head.metadata()["expert"] == "mean_rich_stats_residual"
    assert set(dict(head_model.head.named_parameters())) == {
        "baseline.linear.weight",
        "baseline.linear.bias",
        "expert.linear.weight",
        "expert.linear.bias",
        "expert.correction.weight",
        "gate_logit",
    }


def test_wrapper_build_reliability_shrinkage_exposes_train_only_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(
        variant="mean_reliability_shrinkage"
    ).build(encoder, {"eeg": eeg[:1]}, n_outputs=1)
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert head_model.head.metadata()["reliability_features"] == [
        "log1p_dispersion",
        "log1p_mean_token_norm",
        "active_token_fraction",
    ]
    assert head_model.head.metadata()["gate_initialization"] == -4.0
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "correction.weight",
        "gate.weight",
        "gate.bias",
    }


def test_wrapper_build_reliability_stable_reuses_h2_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(
        variant="mean_reliability_stable"
    ).build(encoder, {"eeg": eeg[:1]}, n_outputs=1)
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert head_model.head.__class__.__name__ == "MeanReliabilityShrinkageHead"


def test_h6_training_patch_adds_consistency_and_restores_original_gate() -> None:
    class DummyBrain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = reve.MeanReliabilityShrinkageHead(embed_dim=4, n_outputs=1)
            self.current_epoch = 0
            self.trainer = SimpleNamespace(world_size=1)
            self.logged: dict[str, torch.Tensor] = {}

        def model_forward(self, batch: SimpleNamespace) -> torch.Tensor:
            return self.head(batch.data["neuro"])

        def training_step(self, batch: SimpleNamespace, batch_idx: int) -> torch.Tensor:
            del batch_idx
            return self.model_forward(batch).square().mean()

        def log(self, name: str, value: torch.Tensor, **kwargs: object) -> None:
            del kwargs
            self.logged[name] = value.detach()

    module = DummyBrain()
    with torch.no_grad():
        module.head.gate.weight.fill_(0.25)
        module.head.correction.weight.fill_(0.01)
    batch = SimpleNamespace(data={"neuro": torch.randn(2, 6, 4), "target": torch.zeros(2, 1)})
    original_rng = torch.random.get_rng_state()
    patched: list[dict[str, object]] = []
    runtime._patch_h6_training_step(module, run_seed=33, patched_modules=patched)
    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert "train/gate_consistency" in module.logged
    assert torch.isfinite(module.logged["train/gate_consistency"])
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    assert module.head._last_gate_values is not None
    runtime._restore_h6_training_steps(patched)
    assert "training_step" not in module.__dict__


def test_augmentation_consistency_view_is_deterministic_and_sample_local() -> None:
    neuro = torch.randn(2, 5, 4)
    original = neuro.clone()

    first = runtime.make_augmentation_consistency_view(
        neuro,
        run_seed=33,
        epoch=1,
        batch_idx=2,
        noise_scale=0.01,
    )
    second = runtime.make_augmentation_consistency_view(
        neuro,
        run_seed=33,
        epoch=1,
        batch_idx=2,
        noise_scale=0.01,
    )

    assert first.shape == neuro.shape
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(neuro, original)
    assert not torch.equal(first, neuro)


def test_augmentation_consistency_training_patch_adds_loss_and_restores() -> None:
    class DummyBrain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(4, 1)
            self.current_epoch = 0
            self.trainer = SimpleNamespace(world_size=1)
            self.logged: dict[str, torch.Tensor] = {}

        def model_forward(self, batch: SimpleNamespace) -> torch.Tensor:
            return self.head(batch.data["neuro"].mean(dim=1))

        def training_step(self, batch: SimpleNamespace, batch_idx: int) -> torch.Tensor:
            del batch_idx
            return self.model_forward(batch).square().mean()

        def log(self, name: str, value: torch.Tensor, **kwargs: object) -> None:
            del kwargs
            self.logged[name] = value.detach()

    module = DummyBrain()
    batch = SimpleNamespace(data={"neuro": torch.randn(2, 6, 4), "target": torch.zeros(2, 1)})
    original_neuro = batch.data["neuro"].clone()
    original_rng = torch.random.get_rng_state()
    patched: list[dict[str, object]] = []

    runtime._patch_augmentation_consistency_training_step(
        module,
        run_seed=33,
        lambda_consistency=0.05,
        noise_scale=0.01,
        patched_modules=patched,
    )
    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert "train/augmentation_consistency" in module.logged
    assert torch.isfinite(module.logged["train/augmentation_consistency"])
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    torch.testing.assert_close(batch.data["neuro"], original_neuro)

    runtime._restore_augmentation_consistency_training_steps(patched)
    assert "training_step" not in module.__dict__


def test_h6_noise_view_is_deterministic_and_does_not_change_global_rng() -> None:
    neuro = torch.randn(2, 3, 12)
    before = torch.random.get_rng_state()
    first = runtime.make_h6_training_view(
        neuro, run_seed=33, epoch=1, batch_idx=4
    )
    after = torch.random.get_rng_state()
    second = runtime.make_h6_training_view(
        neuro, run_seed=33, epoch=1, batch_idx=4
    )

    torch.testing.assert_close(before, after)
    torch.testing.assert_close(first, second)
    assert first.shape == neuro.shape
    assert not torch.equal(first, neuro)


def test_h6_gate_consistency_loss_is_finite_and_has_gate_gradients() -> None:
    alpha_one = torch.tensor([[0.1], [0.2]], requires_grad=True)
    alpha_two = torch.tensor([[0.3], [0.4]], requires_grad=True)

    penalty = runtime.h6_gate_consistency_loss(alpha_one, alpha_two)
    assert penalty.item() == pytest.approx(0.00004)
    penalty.backward()
    assert alpha_one.grad is not None
    assert alpha_two.grad is not None
    assert torch.isfinite(alpha_one.grad).all()
    assert torch.isfinite(alpha_two.grad).all()


def test_wrapper_build_grouped_rich_stats_shrinkage_exposes_zero_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(
        variant="grouped_rich_stats_shrinkage"
    ).build(encoder, {"eeg": eeg[:1]}, n_outputs=1)
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert torch.equal(head_model.head.gates, torch.zeros_like(head_model.head.gates))
    assert head_model.head.metadata()["statistic_groups"] == [
        "std",
        "range",
        "mad",
        "mean_abs",
    ]
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "projections.0.weight",
        "projections.0.bias",
        "projections.1.weight",
        "projections.1.bias",
        "projections.2.weight",
        "projections.2.bias",
        "projections.3.weight",
        "projections.3.bias",
        "gates",
    }


def test_wrapper_build_grouped_stats_shared_gate_exposes_one_zero_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(
        variant="grouped_stats_shared_gate"
    ).build(encoder, {"eeg": eeg[:1]}, n_outputs=1)
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert head_model.head.gate.item() == 0.0
    assert head_model.head.metadata()["gate_parameterization"] == "shared_scalar"
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "projections.0.weight",
        "projections.0.bias",
        "projections.1.weight",
        "projections.1.bias",
        "projections.2.weight",
        "projections.2.bias",
        "projections.3.weight",
        "projections.3.bias",
        "gate",
    }


def test_wrapper_build_temporal_pyramid_stats_exposes_zero_start_low_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(
        variant="temporal_pyramid_stats"
    ).build(encoder, {"eeg": eeg[:1]}, n_outputs=1)
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert head_model.head.segments == 2
    assert head_model.head.correction_rank == 8
    assert torch.equal(head_model.head.up.weight, torch.zeros_like(head_model.head.up.weight))
    assert head_model.head.metadata()["low_rank_parameterization"] == "down_then_up"
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "down.weight",
        "up.weight",
    }


def test_wrapper_build_mean_stats_residual_detached_starts_as_mean_linear_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    encoder = _FakeReveWrapper()
    eeg = torch.randn(2, 3, 5)

    head_model = make_upstream_reve_wrapper(variant="mean_stats_residual_detached").build(
        encoder, {"eeg": eeg[:1]}, n_outputs=1
    )
    output = head_model(eeg)

    assert output.shape == (2, 1)
    assert torch.equal(
        head_model.head.correction.weight,
        torch.zeros_like(head_model.head.correction.weight),
    )
    assert set(dict(head_model.head.named_parameters())) == {
        "linear.weight",
        "linear.bias",
        "correction.weight",
    }


def test_mean_linear_copy_build_matches_official_dummy_forward_rng_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_neuralbench(monkeypatch)
    eeg = torch.randn(1, 3, 5)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        encoder = _RngConsumingReveWrapper()
        head_model = make_upstream_reve_wrapper(variant="mean_linear_copy").build(
            encoder, {"eeg": eeg}, n_outputs=1
        )
        actual_weight = head_model.head.linear.weight.detach().clone()
        actual_bias = head_model.head.linear.bias.detach().clone()

        torch.manual_seed(123)
        reference_encoder = _RngConsumingReveWrapper()
        with torch.no_grad():
            reference_encoder(eeg)
        reference_linear = nn.Linear(4, 1)

    torch.testing.assert_close(actual_weight, reference_linear.weight)
    torch.testing.assert_close(actual_bias, reference_linear.bias)


def test_mean_linear_copy_is_local_control_not_official_variant() -> None:
    assert reve.validate_head_variant("mean_linear_copy") == "mean_linear_copy"
    with pytest.raises(ValueError, match="official"):
        reve.validate_official_head_variant("mean_linear_copy")


def test_head_protocol_classification_is_disjoint() -> None:
    assert reve.validate_head_variant("last_tuned") == "last_tuned"
    assert reve.validate_official_head_variant("mean_linear") == "mean_linear"
    assert reve.validate_last_tuned_protocol("last_tuned") == "last_tuned"

    with pytest.raises(ValueError, match="official"):
        reve.validate_official_head_variant("last_tuned")
    with pytest.raises(ValueError, match="last_tuned"):
        reve.validate_last_tuned_protocol("last")


def test_last_tuned_cli_smoke_reports_explicit_finite_smoke_provenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "last_tuned"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "last_tuned"
    assert result["query_initialization"] == "smoke_synthetic_mean_token"
    assert result["query_initialization_provenance"] == "smoke"
    assert result["prediction_finite"] is True
    assert result["metadata_finite"] is True
    assert json.dumps(result)


def test_mean_anchor_cli_smoke_reports_explicit_finite_smoke_provenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_anchor"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_anchor"
    assert result["query_initialization"] == "smoke_synthetic_mean_token"
    assert result["query_initialization_provenance"] == "smoke"
    assert result["prediction_finite"] is True
    assert result["metadata_finite"] is True
    assert json.dumps(result)


def test_mean_residual_cli_smoke_reports_explicit_finite_smoke_provenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_residual"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_residual"
    assert result["query_initialization"] == "smoke_synthetic_mean_token"
    assert result["query_initialization_provenance"] == "smoke"
    assert result["prediction_finite"] is True
    assert result["metadata_finite"] is True
    assert json.dumps(result)


def test_mean_vector_anchor_cli_smoke_reports_explicit_finite_smoke_provenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_vector_anchor"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_vector_anchor"
    assert result["query_initialization"] == "smoke_synthetic_mean_token"
    assert result["query_initialization_provenance"] == "smoke"
    assert result["prediction_finite"] is True
    assert result["metadata_finite"] is True
    assert json.dumps(result)


def test_mean_mlp_residual_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_mlp_residual"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_mlp_residual"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]
    assert json.dumps(result)


def test_mean_stats_residual_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_stats_residual"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_stats_residual"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]
    assert json.dumps(result)


def test_mean_stats_residual_detached_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_stats_residual_detached"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_stats_residual_detached"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]
    assert json.dumps(result)


def test_mean_linear_copy_cli_smoke_reports_no_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_linear_copy"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_linear_copy"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]


def test_grouped_rich_stats_cli_smoke_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "grouped_rich_stats_shrinkage"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "grouped_rich_stats_shrinkage"
    assert result["prediction_shape"] == [2, 1]
    assert result["head_metadata"]["gate_initialization"] == 0.0
    assert result["head_metadata"]["parameter_count"] > 0


def test_reliability_shrinkage_cli_smoke_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_reliability_shrinkage"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_reliability_shrinkage"
    assert result["prediction_shape"] == [2, 1]
    assert result["head_metadata"]["gate_initialization"] == -4.0
    assert result["head_metadata"]["alpha_max"] == 0.5


def test_grouped_stats_shared_gate_cli_smoke_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "grouped_stats_shared_gate"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "grouped_stats_shared_gate"
    assert result["prediction_shape"] == [2, 1]
    assert result["head_metadata"]["gate_initialization"] == 0.0
    assert result["head_metadata"]["gate_parameterization"] == "shared_scalar"


def test_temporal_pyramid_stats_cli_smoke_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "temporal_pyramid_stats"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "temporal_pyramid_stats"
    assert result["prediction_shape"] == [2, 1]
    assert result["head_metadata"]["segments"] == 2
    assert result["head_metadata"]["correction_rank"] == 8
    assert result["head_metadata"]["up_initialization"] == "zero"


def test_mean_covariance_cli_smoke_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_covariance_residual"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_covariance_residual"
    assert result["prediction_shape"] == [2, 1]
    assert result["head_metadata"]["covariance_mode"] == "diagonal"
    assert result["head_metadata"]["projection_rank"] == 4
    assert result["head_metadata"]["up_initialization"] == "zero"


def test_mean_linear_detached_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_linear_detached"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_linear_detached"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]


def test_mean_linear_warmup_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_linear_warmup"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_linear_warmup"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]


def test_mean_linear_gradient_scaled_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_linear_gradient_scaled"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_linear_gradient_scaled"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]


def test_mean_linear_probe_scaled_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_linear_probe_scaled"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_linear_probe_scaled"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]


def test_mean_stats_residual_gradient_scaled_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_stats_residual_gradient_scaled"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_stats_residual_gradient_scaled"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]


def test_mean_stats_probe_scaled_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_stats_probe_scaled"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_stats_probe_scaled"
    assert result["query_initialization"] == "not_applicable"
    assert result["prediction_shape"] == [2, 1]


def test_mean_stats_attention_residual_cli_smoke_reports_finite_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_smoke_stack(monkeypatch)

    assert official.main(["--smoke-head", "mean_stats_attention_residual"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["head_variant"] == "mean_stats_attention_residual"
    assert result["query_initialization"] == "smoke_synthetic_mean_token"
    assert result["prediction_shape"] == [2, 1]


def test_synthetic_reve_smoke_api_still_accepts_explicit_positions() -> None:
    model = _SmokeReve(embed_dim=4, depth=2, n_chans=3, n_times=5)
    eeg = torch.randn(2, 3, 5)
    positions = torch.randn(2, 3, 3)

    output = model(eeg, pos=positions)

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(model.seen_pos, positions)


def test_last_tuned_head_variant_cli_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="last_tuned")

    assert report["head_variant"] == "last_tuned"


def test_mean_linear_copy_cli_report_identifies_local_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_linear_copy")

    assert report["head_variant"] == "mean_linear_copy"
    assert report["head_source"] == "local_mean_linear_copy"
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report.get("protocol_class") != "tuning"


def test_grouped_rich_stats_cli_report_identifies_shrinkage_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(
        monkeypatch, tmp_path, variant="grouped_rich_stats_shrinkage"
    )

    assert report["head_variant"] == "grouped_rich_stats_shrinkage"
    assert report["head_source"] == "local_grouped_rich_stats_shrinkage"
    assert report["head_architecture"] == "mean_grouped_rich_stats_zero_gate_shrinkage"
    assert report["statistic_groups"] == ["std", "range", "mad", "mean_abs"]
    assert report["gate_initialization"] == 0.0
    assert report["projection_initialization"] == (
        "linspace_-1_1_roll_group_plus_row_alternating_sign_l2_normalized_zero_bias"
    )
    assert report["correction_scale"] == 0.5


def test_grouped_stats_shared_gate_cli_report_identifies_shared_gate_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(
        monkeypatch, tmp_path, variant="grouped_stats_shared_gate"
    )

    assert report["head_variant"] == "grouped_stats_shared_gate"
    assert report["head_source"] == "local_grouped_stats_shared_gate"
    assert report["head_architecture"] == "mean_grouped_stats_shared_gate"
    assert report["statistic_groups"] == ["std", "range", "mad", "mean_abs"]
    assert report["gate_initialization"] == 0.0
    assert report["gate_parameterization"] == "shared_scalar"
    assert report["correction_scale"] == 0.5


def test_temporal_pyramid_stats_cli_report_identifies_low_rank_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(
        monkeypatch, tmp_path, variant="temporal_pyramid_stats"
    )

    assert report["head_variant"] == "temporal_pyramid_stats"
    assert report["head_source"] == "local_temporal_pyramid_stats"
    assert report["head_architecture"] == "mean_temporal_pyramid_stats_low_rank_residual"
    assert report["segments"] == 2
    assert report["correction_rank"] == 8
    assert report["low_rank_parameterization"] == "down_then_up"
    assert report["correction_initialization"] == "zero_via_up_factor"


def test_mean_covariance_cli_report_identifies_diagonal_covariance_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(
        monkeypatch, tmp_path, variant="mean_covariance_residual"
    )

    assert report["head_variant"] == "mean_covariance_residual"
    assert report["head_source"] == "local_mean_covariance_residual"
    assert report["head_architecture"] == "mean_diagonal_covariance_low_rank_residual"
    assert report["covariance_mode"] == "diagonal"
    assert report["covariance_features"] == "diagonal_sample_variance"
    assert report["projection_rank"] == 4
    assert report["low_rank_parameterization"] == "down_then_up"
    assert report["correction_initialization"] == "zero_via_up_factor"


def test_mean_linear_detached_cli_report_identifies_frozen_encoder_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_linear_detached")

    assert report["head_variant"] == "mean_linear_detached"
    assert report["head_source"] == "local_mean_linear_detached"
    assert report["head_architecture"] == "mean_linear_detached_encoder"
    assert report["encoder_gradient"] == "detached"
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_linear_warmup_cli_report_identifies_gated_residual_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_linear_warmup")

    assert report["head_variant"] == "mean_linear_warmup"
    assert report["head_source"] == "local_mean_linear_warmup"
    assert report["head_architecture"] == "mean_linear_zero_gate_residual_warmup"
    assert report["gate_initialization"] == 0.0
    assert report["baseline_encoder_gradient"] == "detached"
    assert report["residual_encoder_gradient"] == "enabled_after_gate_update"
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_linear_gradient_scaled_cli_report_identifies_gradient_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_linear_gradient_scaled")

    assert report["head_variant"] == "mean_linear_gradient_scaled"
    assert report["head_source"] == "local_mean_linear_gradient_scaled"
    assert report["head_architecture"] == "mean_linear_gradient_scaled"
    assert report["encoder_gradient_scale"] == 0.1
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_linear_probe_scaled_cli_report_identifies_gradient_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_linear_probe_scaled")

    assert report["head_variant"] == "mean_linear_probe_scaled"
    assert report["head_source"] == "local_mean_linear_probe_scaled"
    assert report["head_architecture"] == "mean_linear_probe_gradient_scaled"
    assert report["encoder_gradient_scale"] == 0.1
    assert report["probe_gradient_scale"] == 10.0
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_stats_residual_gradient_scaled_cli_report_identifies_gradient_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_stats_residual_gradient_scaled")

    assert report["head_variant"] == "mean_stats_residual_gradient_scaled"
    assert report["head_source"] == "local_mean_stats_residual_gradient_scaled"
    assert report["head_architecture"] == "mean_stats_zero_correction_gradient_scaled"
    assert report["encoder_gradient_scale"] == 0.5
    assert report["correction_backbone_gradient"] == "detached"
    assert report["correction_features"] == "per_feature_std_and_range"
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_stats_probe_scaled_cli_report_identifies_frozen_probe_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_stats_probe_scaled")

    assert report["head_variant"] == "mean_stats_probe_scaled"
    assert report["head_source"] == "local_mean_stats_probe_scaled"
    assert report["head_architecture"] == "mean_stats_zero_correction_probe_gradient_scaled"
    assert report["encoder_gradient_scale"] == 1.0
    assert report["probe_gradient_scale"] == 2.0
    assert report["correction_backbone_gradient"] == "detached"
    assert report["correction_features"] == "per_feature_std_and_range"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_stats_attention_residual_cli_report_identifies_combined_correction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_stats_attention_residual")

    assert report["head_variant"] == "mean_stats_attention_residual"
    assert report["head_source"] == "local_mean_stats_attention_residual"
    assert report["head_architecture"] == "mean_stats_attention_zero_correction"
    assert report["head_query_initialization"] == "train_dummy_final_token_mean"
    assert report["attention_correction_scale"] == 0.25
    assert report["stats_correction_scale"] == 0.5
    assert report["correction_initialization"] == "zero"
    assert report.get("protocol_class") != "tuning"


def test_mean_anchor_cli_report_identifies_baseline_preserving_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_anchor")

    assert report["head_variant"] == "mean_anchor"
    assert report["head_source"] == "local_mean_anchor"
    assert report["head_architecture"] == "mean_anchor_train_dummy_query_residual"
    assert report["head_query_initialization"] == "train_dummy_final_token_mean"
    assert report["gamma_initialization"] == 0.0
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_residual_cli_report_identifies_baseline_preserving_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_residual")

    assert report["head_variant"] == "mean_residual"
    assert report["head_source"] == "local_mean_residual"
    assert report["head_architecture"] == "mean_residual_zero_correction_query_attention"
    assert report["head_query_initialization"] == "train_dummy_final_token_mean"
    assert report["correction_initialization"] == "zero"
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_vector_anchor_cli_report_identifies_baseline_preserving_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_vector_anchor")

    assert report["head_variant"] == "mean_vector_anchor"
    assert report["head_source"] == "local_mean_vector_anchor"
    assert report["head_architecture"] == "mean_vector_anchor_train_dummy_query_residual"
    assert report["head_query_initialization"] == "train_dummy_final_token_mean"
    assert report["gamma_initialization"] == 0.0
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_mlp_residual_cli_report_identifies_baseline_preserving_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_mlp_residual")

    assert report["head_variant"] == "mean_mlp_residual"
    assert report["head_source"] == "local_mean_mlp_residual"
    assert report["head_architecture"] == "mean_mlp_zero_correction"
    assert report["head_query_initialization"] == "not_applicable"
    assert report["correction_initialization"] == "zero"
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_stats_residual_cli_report_identifies_baseline_preserving_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_stats_residual")

    assert report["head_variant"] == "mean_stats_residual"
    assert report["head_source"] == "local_mean_stats_residual"
    assert report["head_architecture"] == "mean_stats_zero_correction"
    assert report["head_query_initialization"] == "not_applicable"
    assert report["correction_initialization"] == "zero"
    assert report["correction_features"] == "per_feature_std_and_range"
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_mean_stats_residual_detached_cli_report_identifies_detached_correction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant="mean_stats_residual_detached")

    assert report["head_variant"] == "mean_stats_residual_detached"
    assert report["head_source"] == "local_mean_stats_residual_detached"
    assert report["head_architecture"] == "mean_stats_zero_correction_detached_statistics"
    assert report["correction_backbone_gradient"] == "detached"
    assert report["correction_initialization"] == "zero"
    assert report["correction_features"] == "per_feature_std_and_range"
    assert report["head_linear_initialization"] == "torch_nn_linear_default"
    assert report["normalization"] == "none"
    assert report.get("protocol_class") != "tuning"


def test_last_tuned_cli_report_contains_tuning_metadata_and_selection_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_kwargs: dict[str, object] = {}
    fake_groups = [
        {
            "name": "base",
            "parameter_names": [
                "encoder.backbone.weight",
                "head.rms_norm.weight",
                "head.linear.weight",
                "head.linear.bias",
                "head.gate_logit",
            ],
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
        },
        {
            "name": "query",
            "parameter_names": ["head.query_token"],
            "learning_rate": 1e-5,
            "weight_decay": 0.05,
        },
    ]
    report = _run_fake_main_report(
        monkeypatch,
        tmp_path,
        variant="last_tuned",
        official_results=[
            {
                "validation_pearsonr": 0.42,
                "test/pearsonr": 0.17,
                "tuning_metadata": {
                    "head_source": "upstream_reve_tuned",
                    "head_architecture": "last_tuned_residual_query_attention",
                    "protocol_class": "tuning",
                    "residual_initial_alpha": 0.1,
                    "query_initialization": "train_dummy_final_token_mean",
                    "query_initialization_input_sha256": "a" * 64,
                    "optimizer": "AdamW",
                    "base_learning_rate": 1e-4,
                    "query_learning_rate": 1e-5,
                    "scheduler": "OneCycleLR",
                    "weight_decay": 0.05,
                    "scheduler_max_lr": [1e-4, 1e-5],
                    "scheduler_pct_start": 0.1,
                    "scheduler_anneal_strategy": "cos",
                    "scheduler_div_factor": 25.0,
                    "scheduler_final_div_factor": 1e4,
                    "scheduler_interval": "step",
                    "scheduler_frequency": 1,
                    "optimizer_param_groups": fake_groups,
                    "monitor": "val/pearsonr",
                    "test_pearsonr_role": "diagnostic_only",
                },
            }
        ],
        captured_kwargs=captured_kwargs,
)

    assert captured_kwargs["head_variant"] == "last_tuned"
    assert captured_kwargs["seeds"] == (33,)
    assert captured_kwargs["manifest_path"] == tmp_path / "manifest.csv"
    assert captured_kwargs["data_root"] == tmp_path / "data"
    assert captured_kwargs["config_path"] == (
        tmp_path / "results" / "last_tuned" / "seed33" / "neuralbench_config.json"
)
    config_payload = json.loads(Path(captured_kwargs["config_path"]).read_text(encoding="utf-8"))
    assert config_payload["DATA_DIR"] == str(tmp_path / "data")
    assert config_payload["SAVE_DIR"] == str(tmp_path / "results" / "last_tuned" / "seed33")
    run_metadata = captured_kwargs["run_metadata"]
    assert run_metadata["head_variant"] == "last_tuned"
    assert run_metadata["head_source"] == "upstream_reve_tuned"
    assert run_metadata["head_query_initialization"] == (
        "train_dummy_final_token_mean"
)
    assert run_metadata["protocol"]["monitor"] == "val/pearsonr"
    assert run_metadata["seed"] == 33
    assert run_metadata["data_seed"] == 33

    assert report["head_variant"] == "last_tuned"
    assert report["head_source"] == "upstream_reve_tuned"
    assert report["head_architecture"] == "last_tuned_residual_query_attention"
    assert report["protocol_class"] == "tuning"
    assert report["residual_initial_alpha"] == pytest.approx(0.1)
    assert report["query_initialization"] == "train_dummy_final_token_mean"
    assert len(report["query_initialization_input_sha256"]) == 64
    assert report["query_learning_rate"] == pytest.approx(1e-5)
    assert report["base_learning_rate"] == pytest.approx(1e-4)
    assert report["optimizer"] == "AdamW"
    assert report["weight_decay"] == pytest.approx(0.05)
    assert report["scheduler"] == "OneCycleLR"
    assert report["scheduler_max_lr"] == pytest.approx([1e-4, 1e-5])
    assert report["scheduler_pct_start"] == pytest.approx(0.1)
    assert report["scheduler_anneal_strategy"] == "cos"
    assert report["scheduler_div_factor"] == pytest.approx(25.0)
    assert report["scheduler_final_div_factor"] == pytest.approx(1e4)
    assert report["scheduler_interval"] == "step"
    assert report["scheduler_frequency"] == 1
    assert report["optimizer_param_groups"] == fake_groups
    assert report["optimizer_param_groups"][0]["parameter_names"] == [
        "encoder.backbone.weight",
        "head.rms_norm.weight",
        "head.linear.weight",
        "head.linear.bias",
        "head.gate_logit",
    ]
    assert set(report["optimizer_param_groups"][1]["parameter_names"]) == {
        "head.query_token"
    }
    assert "head.query_token" not in set(report["optimizer_param_groups"][0]["parameter_names"])
    assert report["monitor"] == "val/pearsonr"
    assert report["test_pearsonr_role"] == "diagnostic_only"
    assert report["official_results"][0]["test/pearsonr"] == pytest.approx(0.17)


def test_checkpoint_selection_ignores_higher_test_pearson(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    experiment = _last_tuned_experiment_config()
    optimizer_config = _last_tuned_optimizer_config()

    # This is the production protocol gate: it validates the resolved config
    # and optimizer metadata before any local selector demonstration below.
    assert (
        reve.validate_last_tuned_protocol("last_tuned", experiment=experiment, optimizer_config=optimizer_config)
        == "last_tuned"
)
    assert experiment.trainer_config.monitor == "val/pearsonr"
    assert experiment.checkpoint_selection.monitor == "val/pearsonr"
    assert experiment.checkpoint_selection.mode == "max"
    assert experiment.checkpoint_selection.test_pearsonr_role == "diagnostic_only"

    invalid_checkpoint_experiment = _last_tuned_experiment_config()
    invalid_checkpoint_experiment.checkpoint_selection.monitor = "test/pearsonr"
    with pytest.raises((ProtocolMismatchError, ValueError)):
        reve.validate_last_tuned_protocol("last_tuned", experiment=invalid_checkpoint_experiment, optimizer_config=_last_tuned_optimizer_config())

    records = [
        {
            "epoch": 1,
            "checkpoint": "epoch=1-val=0.90.ckpt",
            "pearsonr": 0.90,
            "test_pearsonr": 0.10,
        },
        {
            "epoch": 2,
            "checkpoint": "epoch=2-val=0.80.ckpt",
            "pearsonr": 0.80,
            "test_pearsonr": 0.99,
        },
    ]

    class FakeValidationCheckpointSelector:
        def __init__(self, monitor: str) -> None:
            self.monitor = monitor

        def select(self, candidates: list[dict[str, object]]) -> dict[str, object]:
            metric = self.monitor.removeprefix("val/")
            return max(candidates, key=lambda row: float(row[metric]))

    selector = FakeValidationCheckpointSelector("val/pearsonr")
    selected = selector.select(records)
    selected_with_changed_test = selector.select([{**records[0], "test_pearsonr": 0.01}, {**records[1], "test_pearsonr": 1.00},])

    assert selected["checkpoint"] == "epoch=1-val=0.90.ckpt"
    assert selected_with_changed_test["checkpoint"] == selected["checkpoint"]

    captured_kwargs: dict[str, object] = {}
    report = _run_fake_main_report(
        monkeypatch,
        tmp_path,
        variant="last_tuned",
        official_results=[
            {
                "validation_pearsonr": selected["pearsonr"],
                "test_pearsonr": selected["test_pearsonr"],
                "epoch_metrics": records,
            }
        ],
        captured_kwargs=captured_kwargs,
)
    assert report["protocol"]["monitor"] == "val/pearsonr"
    assert captured_kwargs["run_metadata"]["protocol"]["monitor"] == "val/pearsonr"
    assert report["checkpoint_selection_monitor"] == "val/pearsonr"
    assert report["selected_checkpoint_epoch"] == 1
    assert report["test_pearsonr_role"] == "diagnostic_only"


@pytest.mark.parametrize("variant", reve.OFFICIAL_HEAD_VARIANTS)
def test_official_cli_reports_remain_non_tuning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variant: str,
) -> None:
    report = _run_fake_main_report(monkeypatch, tmp_path, variant=variant)

    assert report["head_variant"] == variant
    assert report.get("protocol_class") != "tuning"
    assert report["head_source"] == (
        "neuralbench_default" if variant == "mean_linear" else "upstream_reve"
)
    assert "query_learning_rate" not in report
    assert "optimizer_param_groups" not in report


def test_protocol_validator_accepts_canonical_config() -> None:
    validate_official_protocol(_protocol_config(), n_total_params=10, n_trainable_params=10)


def test_protocol_path_reads_pydantic_discriminator_from_model_dump() -> None:
    class DiscriminatedConfig:
        def model_dump(self) -> dict[str, str]:
            return {"name": "AdamW"}

    config = SimpleNamespace(optimizer=DiscriminatedConfig())

    assert _get_path(config, "optimizer.name") == "AdamW"


def test_protocol_validator_rejects_any_locked_mismatch() -> None:
    config = _protocol_config()
    config.trainer_config.n_epochs = 2

    with pytest.raises(ProtocolMismatchError, match="n_epochs"):
        validate_official_protocol(config)


def test_seed_routing_is_explicit_and_unique() -> None:
    assert validate_seeds((33, 34, 35)) == (33, 34, 35)

    with pytest.raises(ValueError, match="unique"):
        validate_seeds((33, 33))


def test_strict_validation_only_is_the_safe_evaluation_default() -> None:
    assert validate_evaluation_options("strict", strict_final_test=False) == (
        "strict",
        False,
    )


def test_legacy_rejects_the_strict_final_test_gate() -> None:
    assert validate_evaluation_options("legacy", strict_final_test=False) == (
        "legacy",
        False,
    )
    with pytest.raises(ValueError, match="strict-final-test"):
        validate_evaluation_options("legacy", strict_final_test=True)


def test_unknown_evaluation_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="evaluation protocol"):
        validate_evaluation_options("diagnostic", strict_final_test=False)


def test_validation_callback_records_only_post_train_validation(tmp_path: Path) -> None:
    output = tmp_path / "epoch_validation_metrics.jsonl"
    callback = EpochValidationMetrics(output, seed=33)
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=0,
        callback_metrics={"val/pearsonr": torch.tensor(0.625)},
    )

    callback.on_validation_epoch_end(trainer, SimpleNamespace())
    assert not output.exists()

    callback.on_train_start(trainer, SimpleNamespace())
    callback.on_validation_epoch_end(trainer, SimpleNamespace())

    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["seed"] == 33
    assert record["epoch"] == 1
    assert record["val/pearsonr"] == pytest.approx(0.625)


def test_validation_callback_has_no_test_loader_reference(tmp_path: Path) -> None:
    callback = EpochValidationMetrics(
        tmp_path / "epoch_validation_metrics.jsonl",
        seed=33,
    )

    assert "test_loader" not in vars(callback)
    assert not any("test" in name for name in vars(callback))


def test_train_age_reference_exporter_uses_training_batches_only(tmp_path: Path) -> None:
    callback = official.TrainAgeReferenceExporter(
        tmp_path / "analysis" / "train_age_reference.jsonl",
        seed=33,
    )
    trainer = SimpleNamespace(sanity_checking=False)
    module = SimpleNamespace()
    callback.on_train_start(trainer, module)
    callback.on_train_batch_end(
        trainer,
        module,
        None,
        SimpleNamespace(
            data={
                "subject_ids": ["s-2", "s-1"],
                "target": torch.tensor([42.0, 31.0]),
            }
        ),
        0,
    )
    callback.on_train_end(trainer, module)

    rows = [
        json.loads(line)
        for line in (tmp_path / "analysis" / "train_age_reference.jsonl").read_text().splitlines()
    ]
    assert [row["subject_id"] for row in rows] == ["s-1", "s-2"]
    assert [row["true_age"] for row in rows] == [31.0, 42.0]
    assert all(row["split"] == "train" for row in rows)


def test_optimizer_evidence_exporter_records_constructed_optimizer(tmp_path: Path) -> None:
    parameter = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.SGD([parameter], lr=0.01, weight_decay=0.1)
    trainer = SimpleNamespace(optimizers=[optimizer], lr_scheduler_configs=[])

    official.OptimizerEvidenceExporter(tmp_path / "optimizer.json").on_fit_start(
        trainer,
        SimpleNamespace(),
    )

    payload = json.loads((tmp_path / "optimizer.json").read_text())
    assert payload["status"] == "complete"
    assert payload["optimizers"][0]["class"] == "SGD"
    assert payload["optimizers"][0]["param_groups"][0]["parameter_count"] == 2


def test_reliability_gate_audit_callback_writes_validation_and_training_summary(
    tmp_path: Path,
) -> None:
    class ReplayModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Module()
            self.head._last_gate_values = torch.tensor([0.1, 0.2])

        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

        def model_forward(self, batch: object) -> None:
            subject_ids = batch.data["subject_ids"]
            value = 0.1 if int(subject_ids[0]) == 7 else 0.4
            self.head._last_gate_values = torch.tensor([value, value])

    pl_module = ReplayModule()
    pl_module.train()
    callback = official.ReliabilityGateAudit(
        tmp_path / "gate_validation_audit.jsonl",
        seed=33,
    )
    trainer = SimpleNamespace(sanity_checking=False, current_epoch=0)
    train_batch = SimpleNamespace(data={"subject_ids": torch.tensor([7, 7])})
    second_train_batch = SimpleNamespace(data={"subject_ids": torch.tensor([8, 8])})
    validation_batch = SimpleNamespace(data={})

    callback.on_train_start(trainer, pl_module)
    callback.on_train_epoch_start(trainer, pl_module)
    callback.on_train_batch_end(trainer, pl_module, None, train_batch, 0)
    pl_module.head._last_gate_values = torch.tensor([0.2, 0.2])
    callback.on_train_batch_end(trainer, pl_module, None, second_train_batch, 1)
    pl_module.head._last_gate_values = torch.tensor([0.3, 0.4])
    callback.on_validation_batch_end(trainer, pl_module, None, validation_batch, 0)
    callback.on_validation_epoch_end(trainer, pl_module)

    record = json.loads((tmp_path / "gate_validation_audit.jsonl").read_text().strip())
    assert record["validation_gate_mean"] == pytest.approx(0.35)
    assert record["training_eta_squared"] == pytest.approx(1.0)
    assert record["training_eta_valid"] is True
    assert record["training_fixed_state_replay"] is True
    assert pl_module.training is True
    assert record["validation"]["sample_count"] == 2


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_validation_callback_rejects_missing_or_non_finite_metric(
    tmp_path: Path,
    value: float | None,
) -> None:
    callback = EpochValidationMetrics(
        tmp_path / "epoch_validation_metrics.jsonl",
        seed=33,
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=0,
        callback_metrics={"val/pearsonr": value} if value is not None else {},
    )
    callback.on_train_start(trainer, SimpleNamespace())

    with pytest.raises(RuntimeError, match="val/pearsonr"):
        callback.on_validation_epoch_end(trainer, SimpleNamespace())


@pytest.mark.parametrize("checkpoint_epoch", [0, 4])
def test_strict_selection_maps_raw_checkpoint_epoch_to_one_based_validation(
    tmp_path: Path,
    checkpoint_epoch: int,
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"epoch": checkpoint_epoch}, checkpoint)
    history = tmp_path / "epoch_validation_metrics.jsonl"
    history.write_text(
        json.dumps(
            {
                "seed": 33,
                "epoch": checkpoint_epoch + 1,
                "val/pearsonr": 0.61,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    selected = runtime._resolve_selected_validation(checkpoint, history, seed=33)

    assert selected["checkpoint_epoch_zero_based"] == checkpoint_epoch
    assert selected["selected_epoch"] == checkpoint_epoch + 1
    assert selected["selected_val_pearsonr"] == pytest.approx(0.61)


def test_strict_selection_rejects_missing_validation_record(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"epoch": 1}, checkpoint)
    history = tmp_path / "epoch_validation_metrics.jsonl"
    history.write_text(
        json.dumps({"seed": 33, "epoch": 1, "val/pearsonr": 0.61}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exactly one validation record"):
        runtime._resolve_selected_validation(checkpoint, history, seed=33)


def test_strict_selection_rejects_duplicate_matching_validation_records(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"epoch": 1}, checkpoint)
    history = tmp_path / "epoch_validation_metrics.jsonl"
    history.write_text(
        "\n".join(
            [
                json.dumps({"seed": 33, "epoch": 2, "val/pearsonr": 0.61}),
                json.dumps({"seed": 33, "epoch": 2, "val/pearsonr": 0.62}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicate epoch"):
        runtime._resolve_selected_validation(checkpoint, history, seed=33)


def _strict_artifact_fixture(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"epoch": 0}, checkpoint)
    config = tmp_path / "config.yaml"
    config.write_text("seed: 33\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("subject,split\nS001,train\n", encoding="utf-8")
    history = tmp_path / "epoch_validation_metrics.jsonl"
    history.write_text(
        json.dumps({"seed": 33, "epoch": 1, "val/pearsonr": 0.61}) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "config": config,
        "manifest": manifest,
        "history": history,
    }


def test_build_strict_selection_binds_all_source_artifacts(tmp_path: Path) -> None:
    paths = _strict_artifact_fixture(tmp_path)

    record = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=paths["manifest"],
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=True,
        sha256_file=official._sha256_file,
    )

    assert record["selected_epoch"] == record["checkpoint_epoch_zero_based"] + 1
    assert record["checkpoint_sha256"] == official._sha256_file(paths["checkpoint"])
    assert record["official_config_sha256"] == official._sha256_file(paths["config"])
    assert record["manifest_sha256"] == official._sha256_file(paths["manifest"])
    assert record["validation_history_sha256"] == official._sha256_file(paths["history"])
    assert record["test_status"] == "sealed"


def test_selective_strict_selection_binds_acquisition_snapshot(
    tmp_path: Path,
) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    acquisition = tmp_path / "selective_task_provenance.json"
    acquisition.write_text('{"data_mode":"selective_task"}\n', encoding="utf-8")
    acquisition_digest = official._sha256_file(acquisition)
    acquisition.with_suffix(".sha256").write_text(acquisition_digest + "\n", encoding="ascii")
    timeline = tmp_path / "selective_task_timeline_provenance.json"
    timeline.write_text(
        json.dumps(
            {
                "data_mode": "selective_task",
                "task": SELECTIVE_TASK,
                "timeline_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        provenance_path=timeline,
        acquisition_provenance_path=acquisition,
        acquisition_provenance_sha256=acquisition_digest,
        data_mode="selective_task",
        timeline_count=1,
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=False,
        sha256_file=official._sha256_file,
    )

    assert record["data_mode"] == "selective_task"
    assert record["acquisition_provenance_path"] == str(acquisition.resolve())
    assert record["acquisition_provenance_sha256"] == acquisition_digest

    selection_path = tmp_path / "selection.json"
    runtime._publish_json_create_if_absent(selection_path, record)
    report = official._strict_report_fields(
        selection_path=selection_path,
        results=[],
        strict_final_test=False,
    )
    assert report["acquisition_provenance_path"] == str(acquisition.resolve())
    assert report["acquisition_provenance_sha256"] == acquisition_digest

    acquisition.with_suffix(".sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="sidecar"):
        runtime._build_strict_selection_record(
            checkpoint_path=paths["checkpoint"],
            official_config_path=paths["config"],
            provenance_path=timeline,
            acquisition_provenance_path=acquisition,
            acquisition_provenance_sha256=acquisition_digest,
            data_mode="selective_task",
            timeline_count=1,
            validation_history_path=paths["history"],
            selection_monitor="val/pearsonr",
            selection_mode="max",
            seed=33,
            head_variant="mean_linear",
            strict_final_test=False,
            sha256_file=official._sha256_file,
        )


def test_strict_report_uses_immutable_config_snapshot(tmp_path: Path) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    selection = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=paths["manifest"],
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=False,
        sha256_file=official._sha256_file,
    )
    paths["config"].write_text("upstream mutated config\n", encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    runtime._publish_json_create_if_absent(selection_path, selection)

    report = official._strict_report_fields(
        selection_path=selection_path,
        results=[],
        strict_final_test=False,
    )

    assert report["official_config_sha256"] == selection["official_config_sha256"]
    assert Path(selection["official_config_path"]).name == "strict_provenance_config.yaml"


def test_publish_json_create_if_absent_is_idempotent_but_never_overwrites(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.json"
    payload = {"value": 1, "name": "winner"}

    runtime._publish_json_create_if_absent(path, payload)
    original = path.read_bytes()
    runtime._publish_json_create_if_absent(path, payload)
    assert path.read_bytes() == original

    with pytest.raises(RuntimeError, match="differs"):
        runtime._publish_json_create_if_absent(path, {"value": 2})


def test_publish_json_create_if_absent_has_one_concurrent_winner(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"

    def publish(value: int) -> str:
        try:
            runtime._publish_json_create_if_absent(path, {"value": value})
            return "ok"
        except RuntimeError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))

    assert sorted(outcomes) == ["lost", "ok"]
    assert json.loads(path.read_text(encoding="utf-8"))["value"] in {1, 2}


def test_test_start_marker_is_persistent_and_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "test_started.json"
    payload = {"selection_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}

    runtime._create_test_start_marker(path, payload)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="already exists"):
        runtime._create_test_start_marker(path, payload)
    assert path.read_bytes() == original


def test_test_start_marker_has_one_concurrent_winner(tmp_path: Path) -> None:
    path = tmp_path / "test_started.json"

    def create(value: int) -> str:
        try:
            runtime._create_test_start_marker(path, {"value": value})
            return "ok"
        except RuntimeError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(create, (1, 2)))

    assert sorted(outcomes) == ["lost", "ok"]


def test_extract_official_test_pearson_requires_exact_single_official_result() -> None:
    assert runtime._extract_official_test_pearson([{"test/pearsonr": 0.61}]) == pytest.approx(0.61)

    with pytest.raises(RuntimeError, match="exactly one"):
        runtime._extract_official_test_pearson([])
    with pytest.raises(RuntimeError, match="exactly one"):
        runtime._extract_official_test_pearson([{"test/pearsonr": 0.61}, {"test/pearsonr": 0.62}])
    with pytest.raises(RuntimeError, match="test/pearsonr"):
        runtime._extract_official_test_pearson([{"test_pearsonr": 0.61}])


def test_runtime_facade_threads_strict_evaluation_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_runtime_run(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(official._runtime, "run_official_subset", fake_runtime_run)
    official.run_official_subset(
        manifest_path=tmp_path / "manifest.csv",
        data_root=tmp_path / "data",
        epoch_metrics_path=tmp_path / "epoch_validation_metrics.jsonl",
        selection_path=tmp_path / "selection.json",
        config_path=tmp_path / "config.json",
        evaluation_protocol="strict",
        strict_final_test=True,
    )

    assert captured["evaluation_protocol"] == "strict"
    assert captured["strict_final_test"] is True
    assert captured["selection_path"] == tmp_path / "selection.json"


def test_runtime_callback_routing_keeps_test_loader_out_of_strict_mode(
    tmp_path: Path,
) -> None:
    class ForbiddenLoaders(dict):
        def __getitem__(self, key: object) -> object:
            if key == "test":
                raise AssertionError("strict callback routing touched test loader")
            return super().__getitem__(key)

    strict_trainer = SimpleNamespace(callbacks=[])
    runtime._append_evaluation_callback(
        strict_trainer,
        evaluation_protocol="strict",
        epoch_metrics_path=tmp_path / "epoch_validation_metrics.jsonl",
        seed=33,
        loaders=ForbiddenLoaders(),
        hooks=official,
    )
    assert isinstance(strict_trainer.callbacks[0], EpochValidationMetrics)

    legacy_trainer = SimpleNamespace(callbacks=[])
    test_loader = object()
    runtime._append_evaluation_callback(
        legacy_trainer,
        evaluation_protocol="legacy",
        epoch_metrics_path=tmp_path / "epoch_test_metrics.jsonl",
        seed=33,
        loaders={"test": test_loader},
        hooks=official,
    )
    assert isinstance(legacy_trainer.callbacks[0], official.EpochTestPearson)
    assert legacy_trainer.callbacks[0].test_loader is test_loader


def test_validation_only_strict_test_phase_seals_selection_without_calling_test(
    tmp_path: Path,
) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    selection = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=paths["manifest"],
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=False,
        sha256_file=official._sha256_file,
    )
    calls: list[str] = []

    def original_test(*_: object) -> dict[str, float]:
        calls.append("test")
        raise AssertionError("validation-only strict phase opened test")

    result = runtime._run_strict_test_phase(
        original_test=original_test,
        experiment=object(),
        loaders={"test": object()},
        best_model_path=str(paths["checkpoint"]),
        selection_path=tmp_path / "selection.json",
        test_started_path=tmp_path / "test_started.json",
        test_completed_path=tmp_path / "test_completed.json",
        selection_record=selection,
        strict_final_test=False,
        invocation_key=1,
        test_invocations=set(),
        sha256_file=official._sha256_file,
    )

    assert result == {}
    assert calls == []
    assert json.loads((tmp_path / "selection.json").read_text())["test_status"] == "withheld"
    assert not (tmp_path / "test_started.json").exists()


def test_final_strict_test_phase_calls_test_once_and_writes_completion_marker(
    tmp_path: Path,
) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    selection = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=paths["manifest"],
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=True,
        sha256_file=official._sha256_file,
    )
    calls: list[str] = []

    def original_test(*_: object) -> dict[str, float]:
        calls.append("test")
        return {"test/pearsonr": 0.6125, "test/mse": 0.2}

    invocation_set: set[int] = set()
    kwargs = {
        "original_test": original_test,
        "experiment": object(),
        "loaders": {"test": object()},
        "best_model_path": str(paths["checkpoint"]),
        "selection_path": tmp_path / "selection.json",
        "test_started_path": tmp_path / "test_started.json",
        "test_completed_path": tmp_path / "test_completed.json",
        "selection_record": selection,
        "strict_final_test": True,
        "invocation_key": 1,
        "test_invocations": invocation_set,
        "sha256_file": official._sha256_file,
    }

    result = runtime._run_strict_test_phase(**kwargs)

    assert result == {"test/pearsonr": 0.6125, "test/mse": 0.2}
    assert calls == ["test"]
    assert (tmp_path / "test_started.json").exists()
    completed = json.loads((tmp_path / "test_completed.json").read_text())
    assert completed["test_pearsonr"] == pytest.approx(0.6125)
    assert completed["test_evaluations"] == 1

    with pytest.raises(RuntimeError, match="already evaluated"):
        runtime._run_strict_test_phase(**kwargs)
    assert calls == ["test"]


def test_final_strict_test_phase_detects_checkpoint_mutation(tmp_path: Path) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    selection = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=paths["manifest"],
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=True,
        sha256_file=official._sha256_file,
    )

    def original_test(*_: object) -> dict[str, float]:
        paths["checkpoint"].write_bytes(b"mutated")
        return {"test/pearsonr": 0.61}

    with pytest.raises(RuntimeError, match="checkpoint hash changed"):
        runtime._run_strict_test_phase(
            original_test=original_test,
            experiment=object(),
            loaders={"test": object()},
            best_model_path=str(paths["checkpoint"]),
            selection_path=tmp_path / "selection.json",
            test_started_path=tmp_path / "test_started.json",
            test_completed_path=tmp_path / "test_completed.json",
            selection_record=selection,
            strict_final_test=True,
            invocation_key=1,
            test_invocations=set(),
            sha256_file=official._sha256_file,
        )
    assert not (tmp_path / "test_completed.json").exists()


def test_strict_test_phase_rejects_prior_legacy_test_artifacts(tmp_path: Path) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    selection = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=paths["manifest"],
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=True,
        sha256_file=official._sha256_file,
    )
    (tmp_path / "epoch_test_metrics.jsonl").write_text(
        '{"seed": 33, "epoch": 1, "test/pearsonr": 0.9}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="legacy test metrics"):
        runtime._run_strict_test_phase(
            original_test=lambda *args: {"test/pearsonr": 0.61},
            experiment=object(),
            loaders={"test": object()},
            best_model_path=str(paths["checkpoint"]),
            selection_path=tmp_path / "selection.json",
            test_started_path=tmp_path / "test_started.json",
            test_completed_path=tmp_path / "test_completed.json",
            selection_record=selection,
            strict_final_test=True,
            invocation_key=1,
            test_invocations=set(),
            sha256_file=official._sha256_file,
        )
    assert not (tmp_path / "selection.json").exists()


def _fake_main_args(tmp_path: Path, *extra: str) -> list[str]:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic\n", encoding="utf-8")
    return [
        "--manifest",
        str(manifest),
        "--data-root",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "results"),
        "--head-variant",
        "mean_linear",
        "--seeds",
        "33",
        *extra,
    ]


def test_cli_defaults_to_strict_validation_only_and_validation_history_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(official, "manifest_sha256", lambda path: "manifest-sha")
    monkeypatch.setattr(
        official,
        "load_manifest_timelines",
        lambda manifest, root: [{"id": "synthetic"}],
    )

    def fake_run(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(official, "run_official_subset", fake_run)
    monkeypatch.setattr(
        official,
        "_strict_report_fields",
        lambda **kwargs: {
            "evaluation_protocol": "strict",
            "strict_final_test": False,
            "test_status": "withheld",
            "test_evaluations": 0,
            "selected_epoch": 1,
            "selected_val_pearsonr": 0.0,
        },
    )

    assert official.main(_fake_main_args(tmp_path)) == 0

    assert captured["evaluation_protocol"] == "strict"
    assert captured["strict_final_test"] is False
    assert Path(captured["epoch_metrics_path"]).name == "epoch_validation_metrics.jsonl"
    assert Path(captured["selection_path"]).name == "selection.json"


def test_cli_rejects_final_test_gate_for_legacy_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(official, "manifest_sha256", lambda path: "manifest-sha")
    monkeypatch.setattr(
        official,
        "load_manifest_timelines",
        lambda manifest, root: [{"id": "synthetic"}],
    )

    with pytest.raises(SystemExit):
        official.main(
            _fake_main_args(
                tmp_path,
                "--evaluation-protocol",
                "legacy",
                "--strict-final-test",
            )
        )


def test_strict_validation_report_contains_no_test_metric(tmp_path: Path) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    selection = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=paths["manifest"],
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=False,
        sha256_file=official._sha256_file,
    )
    selection_path = tmp_path / "selection.json"
    runtime._publish_json_create_if_absent(selection_path, selection)

    report = official._strict_report_fields(
        selection_path=selection_path,
        results=[],
        strict_final_test=False,
    )

    assert report["evaluation_protocol"] == "strict"
    assert report["strict_final_test"] is False
    assert report["test_status"] == "withheld"
    assert report["test_evaluations"] == 0
    assert report["selected_epoch"] == 1
    assert report["selected_val_pearsonr"] == pytest.approx(0.61)
    assert report["validation_metrics"].endswith("epoch_validation_metrics.jsonl")
    assert report["selection_record"].endswith("selection.json")
    assert "test_pearsonr" not in report


def test_strict_final_report_contains_validation_and_test_together(tmp_path: Path) -> None:
    paths = _strict_artifact_fixture(tmp_path)
    selection = runtime._build_strict_selection_record(
        checkpoint_path=paths["checkpoint"],
        official_config_path=paths["config"],
        manifest_path=paths["manifest"],
        validation_history_path=paths["history"],
        selection_monitor="val/pearsonr",
        selection_mode="max",
        seed=33,
        head_variant="mean_linear",
        strict_final_test=True,
        sha256_file=official._sha256_file,
    )
    selection_path = tmp_path / "selection.json"
    start_path = tmp_path / "test_started.json"
    completed_path = tmp_path / "test_completed.json"
    runtime._publish_json_create_if_absent(selection_path, selection)
    selection_sha = official._sha256_file(selection_path)
    runtime._create_test_start_marker(
        start_path,
        {
            "selection_sha256": selection_sha,
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "test_evaluations": 1,
        },
    )
    runtime._create_test_completed_marker(
        completed_path,
        {
            "selection_sha256": selection_sha,
            "checkpoint_sha256_after_test": selection["checkpoint_sha256"],
            "test_pearsonr": 0.61,
            "test_evaluations": 1,
        },
    )

    report = official._strict_report_fields(
        selection_path=selection_path,
        results=[{"test/pearsonr": 0.61}],
        strict_final_test=True,
    )

    assert report["selected_val_pearsonr"] == pytest.approx(0.61)
    assert report["test_pearsonr"] == pytest.approx(0.61)
    assert report["test_evaluations"] == 1
    assert report["checkpoint_integrity_verified"] is True
    assert report["validation_metrics"].endswith("epoch_validation_metrics.jsonl")
    assert report["selection_record"].endswith("selection.json")


def test_strict_summary_aggregates_validation_and_final_test(tmp_path: Path) -> None:
    reports = [
        {
            "evaluation_protocol": "strict",
            "strict_final_test": True,
            "seed": 33,
            "selected_epoch": 1,
            "selected_val_pearsonr": 0.60,
            "test_pearsonr": 0.61,
        },
        {
            "evaluation_protocol": "strict",
            "strict_final_test": True,
            "seed": 34,
            "selected_epoch": 2,
            "selected_val_pearsonr": 0.62,
            "test_pearsonr": 0.63,
        },
    ]

    summary = official._strict_summary_fields(reports)

    assert summary["completed_seed_count"] == 2
    assert summary["selected_epoch_by_seed"] == {"33": 1, "34": 2}
    assert summary["selected_val_pearson_by_seed"] == {"33": 0.60, "34": 0.62}
    assert summary["test_pearson_by_seed"] == {"33": 0.61, "34": 0.63}
    assert summary["test_status"] == "completed"
    assert summary["mean_selected_val_pearson"] == pytest.approx(0.61)
    assert summary["mean_test_pearson"] == pytest.approx(0.62)


def test_strict_summary_validation_only_withholds_test_metrics() -> None:
    summary = official._strict_summary_fields(
        [
            {
                "evaluation_protocol": "strict",
                "strict_final_test": False,
                "seed": 33,
                "selected_epoch": 1,
                "selected_val_pearsonr": 0.60,
            }
        ]
    )

    assert summary["test_status"] == "withheld"
    assert summary["mean_selected_val_pearson"] == pytest.approx(0.60)
    assert not any("test_pearson" in key for key in summary)


def test_strict_summary_keeps_full_data_provenance_by_seed() -> None:
    summary = official._strict_summary_fields(
        [
            {
                "evaluation_protocol": "strict",
                "strict_final_test": True,
                "seed": 33,
                "data_mode": "full",
                "timeline_count": 10,
                "provenance_path": "/runs/seed33/full_data_provenance.json",
                "provenance_sha256": "a" * 64,
                "manifest_path": None,
                "manifest_sha256": None,
                "selected_epoch": 1,
                "selected_val_pearsonr": 0.60,
                "test_pearsonr": 0.61,
            },
            {
                "evaluation_protocol": "strict",
                "strict_final_test": True,
                "seed": 34,
                "data_mode": "full",
                "timeline_count": 10,
                "provenance_path": "/runs/seed34/full_data_provenance.json",
                "provenance_sha256": "b" * 64,
                "manifest_path": None,
                "manifest_sha256": None,
                "selected_epoch": 2,
                "selected_val_pearsonr": 0.62,
                "test_pearsonr": 0.63,
            },
        ]
    )

    assert summary["data_mode"] == "full"
    assert summary["timeline_count"] == 10
    assert summary["provenance_path"] == {
        "33": "/runs/seed33/full_data_provenance.json",
        "34": "/runs/seed34/full_data_provenance.json",
    }
    assert summary["provenance_sha256"] == {"33": "a" * 64, "34": "b" * 64}


def test_strict_summary_keeps_selective_acquisition_by_seed() -> None:
    summary = official._strict_summary_fields(
        [
            {
                "evaluation_protocol": "strict",
                "strict_final_test": True,
                "seed": 33,
                "data_mode": "selective_task",
                "timeline_count": 10,
                "provenance_path": "/runs/seed33/selective_task_timeline_provenance.json",
                "provenance_sha256": "a" * 64,
                "acquisition_provenance_path": "/runs/seed33/selective_task_provenance.json",
                "acquisition_provenance_sha256": "c" * 64,
                "manifest_path": None,
                "manifest_sha256": None,
                "selected_epoch": 1,
                "selected_val_pearsonr": 0.60,
                "test_pearsonr": 0.61,
            },
            {
                "evaluation_protocol": "strict",
                "strict_final_test": True,
                "seed": 34,
                "data_mode": "selective_task",
                "timeline_count": 10,
                "provenance_path": "/runs/seed34/selective_task_timeline_provenance.json",
                "provenance_sha256": "b" * 64,
                "acquisition_provenance_path": "/runs/seed34/selective_task_provenance.json",
                "acquisition_provenance_sha256": "d" * 64,
                "manifest_path": None,
                "manifest_sha256": None,
                "selected_epoch": 2,
                "selected_val_pearsonr": 0.62,
                "test_pearsonr": 0.63,
            },
        ]
    )

    assert summary["data_mode"] == "selective_task"
    assert summary["timeline_count"] == 10
    assert summary["acquisition_provenance_path"] == {
        "33": "/runs/seed33/selective_task_provenance.json",
        "34": "/runs/seed34/selective_task_provenance.json",
    }
    assert summary["acquisition_provenance_sha256_by_seed"] == {
        "33": "c" * 64,
        "34": "d" * 64,
    }


def test_synchronous_aggregator_runner_executes_each_experiment_in_order() -> None:
    calls: list[str] = []

    class FakeExperiment:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self) -> None:
            calls.append(self.name)

    class FakeAggregator:
        experiments = [FakeExperiment("first"), FakeExperiment("second")]

    _run_experiments_synchronously(FakeAggregator())

    assert calls == ["first", "second"]


def test_patch_setup_failure_restores_every_component_patch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_neuralbench = ModuleType("neuralbench")
    fake_data = ModuleType("neuralbench.data")
    fake_main = ModuleType("neuralbench.main")
    fake_cli = ModuleType("neuralbench.cli")
    fake_experiment_config = ModuleType("neuralbench.experiment_config")
    fake_neuralfetch = ModuleType("neuralfetch")
    fake_studies = ModuleType("neuralfetch.studies")
    fake_shirazi = ModuleType("neuralfetch.studies.shirazi2024hbn")

    class FakeData:
        def prepare(self):
            return {}

    class FakeExperiment:
        def _test(self):
            return {}

        def setup_run(self):
            return None

        def prepare_pl_module(self):
            return None

        def setup_trainer(self):
            return None

    class FakeStudy:
        _info = None

        def iter_timelines(self):
            return iter(())

    fake_data.Data = FakeData
    fake_main.Experiment = FakeExperiment
    fake_cli.load_yaml_config = lambda path, *args, **kwargs: {}
    fake_experiment_config.load_yaml_config = lambda path, *args, **kwargs: {}
    fake_shirazi.Shirazi2024Hbn = FakeStudy
    fake_studies.shirazi2024hbn = fake_shirazi
    fake_neuralfetch.studies = fake_studies
    for name, module in {
        "neuralbench": fake_neuralbench,
        "neuralbench.data": fake_data,
        "neuralbench.main": fake_main,
        "neuralbench.cli": fake_cli,
        "neuralbench.experiment_config": fake_experiment_config,
        "neuralfetch": fake_neuralfetch,
        "neuralfetch.studies": fake_studies,
        "neuralfetch.studies.shirazi2024hbn": fake_shirazi,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    Data = FakeData
    Experiment = FakeExperiment
    study = FakeStudy
    original_iter_timelines = study.iter_timelines
    original_info = study._info
    original_prepare = Data.prepare
    original_test = Experiment._test
    original_setup_run = Experiment.setup_run
    original_prepare_pl_module = Experiment.prepare_pl_module
    original_setup_trainer = Experiment.setup_trainer

    class BrokenInfo:
        def model_copy(self, **kwargs: object) -> object:
            raise RuntimeError("synthetic patch setup failure")

    monkeypatch.setattr(official, "load_manifest_timelines", lambda *args: [])
    monkeypatch.setattr(study, "_info", BrokenInfo())

    with pytest.raises(RuntimeError, match="synthetic patch setup failure"):
        official._patch_official_components(
            tmp_path / "manifest.json",
            tmp_path,
            tmp_path / "epoch.jsonl",
            tmp_path / "selection.json",
        )

    assert study.iter_timelines is original_iter_timelines
    assert study._info.__class__ is BrokenInfo
    assert Data.prepare is original_prepare
    assert Experiment._test is original_test
    assert Experiment.setup_run is original_setup_run
    assert Experiment.prepare_pl_module is original_prepare_pl_module
    assert Experiment.setup_trainer is original_setup_trainer
    assert fake_cli.load_yaml_config("grid.yaml") == {}
    assert fake_experiment_config.load_yaml_config("grid.yaml") == {}


def test_restore_official_components_attempts_all_after_one_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[tuple[str, str]] = []

    class RecordingMeta(type):
        def __setattr__(cls, name: str, value: object) -> None:
            attempts.append((cls.__name__, name))
            if cls.__name__ == "FakeData" and name == "prepare":
                raise RuntimeError("synthetic prepare restore failure")
            super().__setattr__(name, value)

    class FakeData(metaclass=RecordingMeta):
        def prepare(self):
            return {}

    class FakeExperiment(metaclass=RecordingMeta):
        def _test(self):
            return {}

        def setup_run(self):
            return None

        def prepare_pl_module(self):
            return None

        def setup_trainer(self):
            return None

    class FakeStudy(metaclass=RecordingMeta):
        _info = None

        def iter_timelines(self):
            return iter(())

    fake_neuralbench = ModuleType("neuralbench")
    fake_data = ModuleType("neuralbench.data")
    fake_main = ModuleType("neuralbench.main")
    fake_cli = ModuleType("neuralbench.cli")
    fake_experiment_config = ModuleType("neuralbench.experiment_config")
    fake_neuralfetch = ModuleType("neuralfetch")
    fake_studies = ModuleType("neuralfetch.studies")
    fake_shirazi = ModuleType("neuralfetch.studies.shirazi2024hbn")
    fake_data.Data = FakeData
    fake_main.Experiment = FakeExperiment
    fake_cli.load_yaml_config = lambda path, *args, **kwargs: {}
    fake_experiment_config.load_yaml_config = lambda path, *args, **kwargs: {}
    fake_shirazi.Shirazi2024Hbn = FakeStudy
    fake_studies.shirazi2024hbn = fake_shirazi
    fake_neuralbench.data = fake_data
    fake_neuralbench.main = fake_main
    fake_neuralbench.cli = fake_cli
    fake_neuralbench.experiment_config = fake_experiment_config
    fake_neuralfetch.studies = fake_studies

    for name, module in {
        "neuralbench": fake_neuralbench,
        "neuralbench.data": fake_data,
        "neuralbench.main": fake_main,
        "neuralbench.cli": fake_cli,
        "neuralbench.experiment_config": fake_experiment_config,
        "neuralfetch": fake_neuralfetch,
        "neuralfetch.studies": fake_studies,
        "neuralfetch.studies.shirazi2024hbn": fake_shirazi,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    originals = {
        "iter_timelines": FakeStudy.iter_timelines,
        "info": FakeStudy._info,
        "prepare": FakeData.prepare,
        "test": FakeExperiment._test,
        "setup_run": FakeExperiment.setup_run,
        "prepare_pl_module": FakeExperiment.prepare_pl_module,
        "setup_trainer": FakeExperiment.setup_trainer,
        "patched_brain_modules": [],
        "cli_loader": (fake_cli, fake_cli.load_yaml_config),
        "experiment_loader": (
            fake_experiment_config,
            fake_experiment_config.load_yaml_config,
),
    }
    monkeypatch.setattr(official, "_restore_last_tuned_configure_optimizers", lambda modules: attempts.append(("tuned", "configure_optimizers")))

    with pytest.raises(RuntimeError, match="official component restoration"):
        official._restore_official_components(originals)

    assert ("FakeData", "prepare") in attempts
    assert ("FakeExperiment", "_test") in attempts
    assert ("FakeExperiment", "setup_run") in attempts
    assert ("FakeExperiment", "prepare_pl_module") in attempts
    assert ("FakeExperiment", "setup_trainer") in attempts
    assert ("FakeStudy", "iter_timelines") in attempts
    assert ("FakeStudy", "_info") in attempts
    assert ("tuned", "configure_optimizers") in attempts


def test_main_failed_run_writes_failure_without_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    data_root = tmp_path / "data"
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.json"
    run_dir = output_dir / "last_tuned" / "seed33"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(official, "manifest_sha256", lambda path: "manifest-sha")
    monkeypatch.setattr(official, "load_manifest_timelines", lambda manifest, root: [{"id": "synthetic"}])

    def fail_run(**kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("synthetic last_tuned failure")

    monkeypatch.setattr(official, "run_official_subset", fail_run)

    result = official.main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
                "--head-variant",
                "last_tuned",
                "--evaluation-protocol",
                "legacy",
                "--seeds",
            "33",
        ]
)

    assert result == 1
    failure_path = run_dir / "failure.json"
    assert failure_path.is_file()
    assert not (run_dir / "report.json").exists()
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "synthetic last_tuned failure"


def test_main_report_write_failure_writes_failure_without_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    data_root = tmp_path / "data"
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.json"
    run_dir = output_dir / "last_tuned" / "seed33"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text("stale\n", encoding="utf-8")

    monkeypatch.setattr(official, "manifest_sha256", lambda path: "manifest-sha")
    monkeypatch.setattr(official, "load_manifest_timelines", lambda manifest, root: [{"id": "synthetic"}])
    monkeypatch.setattr(official, "run_official_subset", lambda **kwargs: [])
    original_write_text = Path.write_text

    def fail_report_write(self, data, *args, **kwargs):
        if self.name == "report.json":
            raise OSError("synthetic report write failure")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_report_write)

    result = official.main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--head-variant",
            "last_tuned",
            "--evaluation-protocol",
            "legacy",
            "--seeds",
            "33",
        ]
)

    assert result == 1
    failure_path = run_dir / "failure.json"
    assert failure_path.is_file()
    assert not (run_dir / "report.json").exists()
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["error_type"] == "OSError"
    assert payload["error"] == "synthetic report write failure"


def test_official_fields_can_be_set_on_exca_frozen_instances() -> None:
    class FrozenObject:
        def __setattr__(self, name: str, value: object) -> None:
            raise RuntimeError("instance is frozen")

    experiment = FrozenObject()

    _set_frozen_experiment_field(experiment, "save_test_predictions", True)

    assert experiment.save_test_predictions is True


def test_source_lock_metadata_is_immutable_and_verifier_rejects_change(
    tmp_path,
    monkeypatch,
) -> None:
    metadata = source_lock_metadata()
    assert metadata["commit"] == UPSTREAM_REVE_COMMIT
    assert set(metadata["file_sha256"]) == set(UPSTREAM_REVE_FILE_HASHES)

    source_root = tmp_path / "reve"
    for relative in UPSTREAM_REVE_FILE_HASHES:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")

    monkeypatch.setattr(
        "neurobench_age.heads.upstream.UPSTREAM_REVE_FILE_HASHES",
        {relative: "bad" for relative in UPSTREAM_REVE_FILE_HASHES},
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_upstream_source_hashes(source_root)


def test_artifact_collector_hashes_config_checkpoint_and_predictions(tmp_path) -> None:
    run_dir = tmp_path / "uid-run"
    prediction_dir = run_dir / "test_predictions"
    prediction_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("seed: 33\n", encoding="utf-8")
    (run_dir / "best.ckpt").write_bytes(b"checkpoint")
    (prediction_dir / "y_pred_0.npy").write_bytes(b"predictions")

    records = collect_run_artifacts(tmp_path)

    assert len(records) == 1
    assert records[0]["selected_checkpoint"]["sha256"]
    assert records[0]["resolved_config"]["sha256"]
    assert records[0]["raw_test_predictions"][0]["size_bytes"] == len(b"predictions")
