import json

import torch
import pytest

from fms_ehrs.framework.training_telemetry import LiveProgressCallback
from fms_ehrs.scripts.train_representation import (
    PeriodicCheckpointCallback,
    RepresentationDataCollator,
    _log_wandb_directory_artifact,
    _latest_complete_checkpoint,
    _validate_checkpoint_contract,
)


def test_representation_data_collator_masks_pad_labels():
    collator = RepresentationDataCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": torch.tensor([5, 6, 0, 0], dtype=torch.long)},
            {"input_ids": torch.tensor([7, 8, 9, 0], dtype=torch.long)},
        ]
    )

    assert batch["input_ids"].tolist() == [[5, 6, 0, 0], [7, 8, 9, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 0]]
    assert batch["labels"].tolist() == [[5, 6, -100, -100], [7, 8, 9, -100]]


def test_best_checkpoint_contract_accepts_matched_epoch_strategies():
    _validate_checkpoint_contract(
        save_strategy="epoch",
        eval_strategy="epoch",
        save_steps=1,
        eval_steps=None,
        load_best_model_at_end=True,
    )


def test_best_checkpoint_contract_rejects_mismatched_strategies():
    with pytest.raises(ValueError, match="matching save_strategy"):
        _validate_checkpoint_contract(
            save_strategy="steps",
            eval_strategy="epoch",
            save_steps=100,
            eval_steps=100,
            load_best_model_at_end=True,
        )


def test_best_checkpoint_contract_rejects_unaligned_step_cadence():
    with pytest.raises(ValueError, match="multiple of eval_steps"):
        _validate_checkpoint_contract(
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=101,
            eval_steps=100,
            load_best_model_at_end=True,
        )


def test_auto_resume_ignores_incomplete_checkpoint(tmp_path):
    completed = tmp_path / "checkpoint-10"
    completed.mkdir()
    (completed / "trainer_state.json").write_text("{}\n")
    (completed / "checkpoint_complete.json").write_text("{}\n")

    incomplete = tmp_path / "checkpoint-20"
    incomplete.mkdir()
    (incomplete / "trainer_state.json").write_text("{}\n")

    assert _latest_complete_checkpoint(tmp_path) == str(completed)


def test_periodic_checkpoint_requests_save_at_interval(monkeypatch):
    callback = PeriodicCheckpointCallback(interval_seconds=600)
    callback._last_checkpoint_at = 0.0
    monkeypatch.setattr(
        "fms_ehrs.scripts.train_representation.time.monotonic",
        lambda: 600.0,
    )

    state = type("State", (), {"global_step": 50})()
    control = type("Control", (), {"should_save": False})()

    callback.on_step_end(None, state, control)

    assert control.should_save


def test_live_progress_callback_writes_local_telemetry(tmp_path):
    callback = LiveProgressCallback(tmp_path, interval_seconds=60)
    state = type(
        "State",
        (),
        {
            "global_step": 0,
            "max_steps": 10,
            "epoch": 0.0,
            "is_world_process_zero": True,
        },
    )()
    control = type("Control", (), {})()

    callback.on_train_begin(None, state, control)

    event = json.loads((tmp_path / "progress.jsonl").read_text().strip())
    assert event["event"] == "train_begin"
    assert event["global_step"] == 0
    assert event["max_steps"] == 10


def test_evaluations_per_epoch_derives_aligned_step_cadence():
    import math

    n_train = 1000
    batch = 4
    accum = 1
    evaluations_per_epoch = 2
    updates_per_epoch = math.ceil(n_train / (batch * accum))
    eval_steps = max(1, math.ceil(updates_per_epoch / evaluations_per_epoch))
    assert updates_per_epoch == 250
    assert eval_steps == 125
    _validate_checkpoint_contract(
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=eval_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
    )


def test_offline_wandb_artifact_is_retained_locally(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")

    _log_wandb_directory_artifact(
        directory=tmp_path,
        artifact_name="checkpoint",
        artifact_type="model",
        metadata={},
        project="test",
        run_name="test",
        require_wandb=True,
    )
