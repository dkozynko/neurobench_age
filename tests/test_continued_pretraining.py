from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from neurobench_age.training.continued_pretraining import (
    ContinuedPretrainingCallback,
    ContinuedPretrainingConfig,
    continued_pretraining_loss,
    mask_neuro_input,
    run_continued_pretraining,
)
import neurobench_age.pipelines.official_runtime as runtime
import neurobench_age.pipelines.official as official


def test_mask_neuro_input_is_deterministic_blockwise_and_non_mutating() -> None:
    neuro = torch.ones(2, 3, 20)
    original = neuro.clone()

    masked_one, mask_one = mask_neuro_input(
        neuro,
        run_seed=33,
        epoch=1,
        batch_idx=2,
        mask_fraction=0.25,
        mask_block_samples=2,
    )
    masked_two, mask_two = mask_neuro_input(
        neuro,
        run_seed=33,
        epoch=1,
        batch_idx=2,
        mask_fraction=0.25,
        mask_block_samples=2,
    )

    assert masked_one.shape == neuro.shape
    assert mask_one.shape == (2, 1, 20)
    assert mask_one.dtype == torch.bool
    assert mask_one.any(dim=-1).all()
    assert torch.equal(mask_one, mask_two)
    torch.testing.assert_close(masked_one, masked_two)
    torch.testing.assert_close(neuro, original)
    torch.testing.assert_close(masked_one.masked_select(~mask_one), neuro.masked_select(~mask_one))
    assert torch.equal(masked_one[:, 0].eq(0), mask_one[:, 0])
    assert torch.equal(masked_one[:, 1].eq(0), mask_one[:, 0])


def test_continued_pretraining_loss_backpropagates_only_through_student() -> None:
    teacher = torch.randn(3, 4)
    student = torch.randn(3, 4, requires_grad=True)

    loss = continued_pretraining_loss(teacher, student)

    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_run_continued_pretraining_uses_batches_without_targets(tmp_path: Path) -> None:
    class DummyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(4, 4)
            self.device = torch.device("cpu")
            self.forward_modes: list[bool] = []

        def model_forward_embedding(self, batch: SimpleNamespace) -> torch.Tensor:
            self.forward_modes.append(bool(self.training))
            return self.encoder(batch.data["neuro"].mean(dim=1))

    module = DummyModule()
    batch = SimpleNamespace(data={"neuro": torch.randn(3, 2, 4)})
    config = ContinuedPretrainingConfig(
        epochs=1,
        mask_fraction=0.25,
        mask_block_samples=1,
        learning_rate=1e-3,
        max_batches=1,
    )

    result = run_continued_pretraining(
        module,
        [batch],
        config=config,
        run_seed=33,
        metadata_path=tmp_path / "continued_pretraining.json",
        metrics_path=tmp_path / "continued_pretraining_metrics.jsonl",
    )

    assert result["batches"] == 1
    assert result["epochs"] == 1
    assert result["age_labels_used"] is False
    assert module.forward_modes == [False, True]
    assert module.training is True
    assert (tmp_path / "continued_pretraining.json").is_file()
    assert (tmp_path / "continued_pretraining_metrics.jsonl").is_file()


def test_callback_is_train_only_and_config_is_serialized(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    official._write_config(
        config_path,
        data_root=tmp_path / "data",
        output_dir=tmp_path / "results",
        head_variant="mean_linear",
        continued_pretraining=True,
        pretraining_epochs=2,
        pretraining_mask_fraction=0.2,
        pretraining_mask_block_samples=10,
        pretraining_learning_rate=2e-5,
        pretraining_weight_decay=0.03,
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["CONTINUED_PRETRAINING"] is True
    assert payload["PRETRAINING_SOURCE_SPLIT"] == "train_only"
    assert payload["PRETRAINING_AGE_LABELS_USED"] is False

    trainer = SimpleNamespace(callbacks=[])
    runtime._append_evaluation_callback(
        trainer,
        evaluation_protocol="strict",
        epoch_metrics_path=tmp_path / "epoch_validation_metrics.jsonl",
        seed=33,
        loaders={"train": object()},
        hooks=official,
        continued_pretraining=True,
        continued_pretraining_config=ContinuedPretrainingConfig(max_batches=1),
    )
    assert isinstance(trainer.callbacks[0], ContinuedPretrainingCallback)
    assert trainer.callbacks[0].train_loader is not None
