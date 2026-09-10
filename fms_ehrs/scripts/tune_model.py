#!/usr/bin/env python3

"""
tune a model with a packing strategy
"""

import importlib.util
import json
import os
import pathlib
import typing
from datetime import datetime, timezone

import fire as fi
import numpy as np
import torch as t
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    LlamaConfig,
    Qwen3Config,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from fms_ehrs.framework.dataset import Datasets
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.training_telemetry import LiveProgressCallback
from fms_ehrs.scripts.train_representation import (
    PeriodicCheckpointCallback,
    SignalCheckpointCallback,
    _latest_complete_checkpoint,
    _mark_checkpoint_complete,
)

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


class CheckpointCompletionCallback(TrainerCallback):
    """Mark regular Trainer checkpoints complete for safe automatic resume."""

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            _mark_checkpoint_complete(
                pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}",
                state,
            )
        return control


def _parse_bool(x: typing.Any, *, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return bool(default)


def _normalize_attn_impl(x: str | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    return s


def _safe_artifact_name(name: str) -> str:
    cleaned = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    out = "".join(cleaned).strip("-")
    return out or "artifact"


def _resolve_latest_checkpoint(output_dir: pathlib.Path) -> pathlib.Path | None:
    """Find latest checkpoint under output_dir or nested run-* dirs."""
    candidates = [
        p
        for p in (
            list(output_dir.glob("checkpoint-*"))
            + list(output_dir.glob("run-*/checkpoint-*"))
        )
        if p.is_dir()
    ]
    if not candidates:
        return None

    def _sort_key(p: pathlib.Path) -> tuple[int, float, str]:
        step = -1
        name = p.name
        if name.startswith("checkpoint-"):
            suffix = name.split("checkpoint-", 1)[1]
            if suffix.isdigit():
                step = int(suffix)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = -1.0
        return (step, mtime, str(p))

    return sorted(candidates, key=_sort_key)[-1]


def _log_wandb_directory_artifact(
    *,
    directory: pathlib.Path,
    artifact_name: str,
    artifact_type: str,
    metadata: dict[str, typing.Any],
    project: str | None,
    run_name: str,
    require_wandb: bool,
) -> None:
    if not directory.exists():
        msg = f"Artifact directory does not exist: {directory}"
        if require_wandb:
            raise RuntimeError(msg)
        logger.warning(msg)
        return

    if str(os.getenv("WANDB_MODE", "")).strip().lower() == "offline":
        logger.info(
            "WANDB_MODE=offline; retaining local artifact %s at %s.",
            artifact_name,
            directory,
        )
        return

    try:
        import wandb

        run = wandb.run
        created_run = False
        if run is None:
            run = wandb.init(
                project=project,
                name=f"{run_name}-artifact",
                job_type="artifact-export",
                reinit=False,
            )
            created_run = True

        artifact = wandb.Artifact(
            name=_safe_artifact_name(artifact_name),
            type=artifact_type,
            metadata=metadata,
        )
        artifact.add_dir(str(directory))
        run.log_artifact(artifact, aliases=["latest"])
        logger.info("Logged W&B artifact '%s' from %s", artifact_name, str(directory))

        if created_run and run is not None:
            run.finish()
    except Exception as e:
        if require_wandb:
            raise RuntimeError(
                f"Failed to log required W&B artifact {artifact_name} from {directory}: {e}"
            ) from e
        logger.warning(
            "W&B artifact upload skipped for %s (%s): %s",
            artifact_name,
            str(directory),
            e,
        )


class NanStoppingCallback(TrainerCallback):
    """stop training on encountering a nan objective"""

    def __init__(self):
        super().__init__()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for k, v in metrics.items():
                if not np.isfinite(v):
                    if state.is_world_process_zero:
                        logger.warning(f"Encountered non-finite metric {k} ({v}).")
                    control.should_training_stop = True


def _with_perplexity(logs: dict[str, float]) -> dict[str, float]:
    enriched = dict(logs)
    for loss_key, perplexity_key in (
        ("loss", "train_perplexity"),
        ("train_loss", "final_train_perplexity"),
        ("eval_loss", "eval_perplexity"),
    ):
        if loss_key not in enriched or perplexity_key in enriched:
            continue
        value = float(enriched[loss_key])
        if np.isfinite(value):
            enriched[perplexity_key] = float(np.exp(min(value, 50.0)))
    return enriched


def _write_loss_history(output_dir: pathlib.Path, log_history: list[dict[str, typing.Any]]) -> None:
    keys = [
        "step",
        "epoch",
        "loss",
        "train_loss",
        "eval_loss",
        "train_perplexity",
        "final_train_perplexity",
        "eval_perplexity",
        "learning_rate",
        "grad_norm",
        "grad_norm_clipped",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_tokens_per_second",
    ]
    rows = [
        {key: _with_perplexity(entry).get(key, "") for key in keys}
        for entry in log_history
        if any(key in entry for key in ("loss", "train_loss", "eval_loss"))
    ]
    with (output_dir / "loss_perplexity_curve.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "loss_perplexity_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class PackedIRBTrainer(Trainer):
    """Trainer for packed token sequences with AdamW via HuggingFace defaults."""

    def log(self, logs: dict[str, float], *args, **kwargs):
        enriched = _with_perplexity(logs)
        if "grad_norm" in enriched:
            enriched["grad_norm_clipped"] = float(
                float(enriched["grad_norm"]) > float(self.args.max_grad_norm)
            )
        return super().log(enriched, *args, **kwargs)


class _PretokenizedProcessingStub:
    """Minimal processing stub for pretokenized `input_ids` datasets."""

    def __init__(self, *, pad_token_id: int, model_max_length: int):
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(pad_token_id)
        self.padding_side = "right"
        self.model_max_length = int(model_max_length)
        self.model_input_names = ["input_ids", "attention_mask"]

    def save_pretrained(self, save_directory: str | os.PathLike):
        out_dir = pathlib.Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "processing_stub.json"
        out_file.write_text(
            json.dumps(
                {
                    "type": "pretokenized_stub",
                    "pad_token_id": self.pad_token_id,
                    "eos_token_id": self.eos_token_id,
                    "padding_side": self.padding_side,
                    "model_max_length": self.model_max_length,
                    "model_input_names": self.model_input_names,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return (str(out_file),)


@logger.log_calls
def main(
    *,
    n_epochs: int = 5,
    max_seq_length: int = int(os.getenv("IRB_MAX_SEQ_LENGTH", "4096")),
    data_version: str = "day_stays",
    model_version: str = "llama1b",
    model_name: str = "meta-llama/Llama-3.2-1B",
    rope_theta: float = 10000.0,
    per_device_train_batch_size: int = int(
        os.getenv("IRB_PER_DEVICE_TRAIN_BATCH_SIZE", "4")
    ),
    per_device_eval_batch_size: int = int(
        os.getenv("IRB_PER_DEVICE_EVAL_BATCH_SIZE", "4")
    ),
    # Policy control:
    # - do_hpo=True: run Optuna HPO (expensive; may exceed cluster walltime)
    # - do_hpo=False: run a single fixed-hyperparameter training
    do_hpo: bool = True,
    # Optional resume path for continuing an interrupted fixed-hyperparameter run.
    resume_from_checkpoint: str | None = os.getenv("IRB_RESUME_FROM_CHECKPOINT", "auto"),
    learning_rate: float = 5e-5,
    lr_min: float = 5e-5,
    lr_max: float = 5e-4,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    logging_steps: int = 100,
    progress_interval_seconds: float = float(
        os.getenv("IRB_PROGRESS_INTERVAL_SECONDS", "60")
    ),
    gradient_accumulation_steps: int = int(
        os.getenv("IRB_GRADIENT_ACCUMULATION_STEPS", "1")
    ),
    # Optional performance settings (objective-preserving):
    # - use_bf16: enable bf16 mixed precision (A100 supports bf16).
    # - attn_implementation: attention backend ("sdpa" or "flash_attention_2").
    use_bf16: bool = _parse_bool(os.getenv("IRB_USE_BF16", "true"), default=True),
    attn_implementation: str | None = _normalize_attn_impl(os.getenv("IRB_ATTN_IMPL", "sdpa")),
    data_dir: os.PathLike = None,
    model_dir: os.PathLike = None,
    collation: typing.Literal["padded", "packed"] = "packed",
    jid: str = os.getenv("SLURM_JOB_ID", ""),
    wandb_project: str = None,
    seed: int = 42,
    n_trials: int = 5,
    save_total_limit: int = 6,
    eval_steps: int | None = None,
    evaluations_per_epoch: int = int(
        os.getenv("IRB_STAGE1_EVALUATIONS_PER_EPOCH", "1")
    ),
    early_stopping_patience: int = int(
        os.getenv("IRB_STAGE1_EARLY_STOPPING_PATIENCE", "3")
    ),
    checkpoint_interval_seconds: float = float(
        os.getenv("IRB_STAGE1_CHECKPOINT_INTERVAL_SECONDS", "600")
    ),
    # Referring to the "Quantifying-Surprise-EHRs" reference implementation:
    # Packed collation is trained using an IterableDataset (no materialization).
    iterable_dataset: bool = True,
    **kwargs,
):
    """pass additional model configuration parameters with kwargs"""
    # Fire may pass booleans as strings (e.g., "false"), so normalize explicitly.
    do_hpo = _parse_bool(do_hpo, default=True)
    iterable_dataset = _parse_bool(iterable_dataset, default=True)
    use_bf16 = _parse_bool(use_bf16, default=True)
    t.manual_seed(seed)
    np.random.seed(seed)
    if isinstance(resume_from_checkpoint, str):
        s = resume_from_checkpoint.strip()
        if s.lower() in ("", "none", "null"):
            resume_from_checkpoint = None
        else:
            resume_from_checkpoint = s

    if int(gradient_accumulation_steps) <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive "
            f"(got {gradient_accumulation_steps})."
        )
    if rope_theta <= 0:
        raise ValueError(f"rope_theta must be positive (got {rope_theta}).")
    if max_grad_norm <= 0:
        raise ValueError(f"max_grad_norm must be positive (got {max_grad_norm}).")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError(f"warmup_ratio must be in [0, 1) (got {warmup_ratio}).")
    if logging_steps <= 0:
        raise ValueError(f"logging_steps must be positive (got {logging_steps}).")
    if progress_interval_seconds <= 0:
        raise ValueError(
            "progress_interval_seconds must be positive "
            f"(got {progress_interval_seconds})."
        )
    if evaluations_per_epoch <= 0:
        raise ValueError(
            f"evaluations_per_epoch must be positive (got {evaluations_per_epoch})."
        )
    if checkpoint_interval_seconds <= 0:
        raise ValueError(
            "checkpoint_interval_seconds must be positive "
            f"(got {checkpoint_interval_seconds})."
        )

    attn_implementation = _normalize_attn_impl(attn_implementation)
    if attn_implementation == "flash_attention_2":
        # Fail early with a clear message if flash-attn isn't installed.
        if importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError(
                "attn_implementation=flash_attention_2 requires the optional `flash-attn` package "
                "(and a compatible GPU/CUDA build)."
            )

    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=model_version, j=jid)
    # In HPO mode, Transformers' W&B callback resolves checkpoint paths relative to
    # `output_dir` while Optuna writes under run-* subdirectories, which can cause
    # "Path is not a directory" on on_save. The same mismatch can occur when
    # resuming from an HPO run-* checkpoint into fixed-training mode.
    # We keep explicit artifact logging below.
    if do_hpo or bool(resume_from_checkpoint):
        os.environ["WANDB_LOG_MODEL"] = "false"
    else:
        os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
    require_wandb = _parse_bool(os.getenv("IRB_REQUIRE_WANDB", "false"), default=False)

    data_dir, model_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (data_dir, model_dir)
    )

    output_dir = model_dir.joinpath("{m}-{j}".format(m=model_version, j=jid))
    output_dir.mkdir(exist_ok=True, parents=True)
    if resume_from_checkpoint == "auto":
        resume_from_checkpoint = _latest_complete_checkpoint(output_dir)
        if resume_from_checkpoint is None:
            logger.info("No completed checkpoint found under %s; starting fresh.", output_dir)
        else:
            logger.info("Auto-resuming from checkpoint: %s", resume_from_checkpoint)

    dataset = Datasets(
        data_version=data_version,
        data_dir=data_dir,
        collation=collation,
        max_seq_length=max_seq_length,
    )

    def model_init(trial=None):
        cfg_kwargs = dict(kwargs)
        if attn_implementation is not None:
            cfg_kwargs["attn_implementation"] = attn_implementation
        cfg_kwargs.setdefault("rope_theta", float(rope_theta))
        model_name_lower = str(model_name).lower()
        if "llama" in model_name_lower:
            config_type = LlamaConfig
            # Llama keeps separate input and output embedding matrices.
            cfg_kwargs.setdefault("tie_word_embeddings", False)
        elif "qwen3" in model_name_lower:
            config_type = Qwen3Config
            # Match the Qwen3-0.6B configuration after reducing attention
            # width: multi-head attention and tied input/output embeddings.
            cfg_kwargs.setdefault(
                "num_key_value_heads",
                int(cfg_kwargs.get("num_attention_heads", 32)),
            )
            cfg_kwargs.setdefault("tie_word_embeddings", True)
        else:
            raise ValueError(
                "ML4H supports only the Llama and Qwen3 backbones; "
                f"received model_name={model_name!r}."
            )
        # The model is initialized from scratch, so use the matching local
        # config and keep training independent of compute-node network access.
        config = config_type(
            vocab_size=len(dataset.vocab),
            bos_token_id=dataset.vocab("TL_START"),
            eos_token_id=dataset.vocab("TL_END"),
            pad_token_id=dataset.vocab("PAD"),
            **cfg_kwargs,
        )
        config._name_or_path = str(model_name)
        logger.info(
            "tie_word_embeddings=%s (True for Qwen, False for Llama)",
            bool(getattr(config, "tie_word_embeddings", False)),
        )
        mdl = AutoModelForCausalLM.from_config(config)
        mdl_params = sum(p.numel() for p in mdl.parameters())
        logger.info("Model initialized, n. param = {}".format(mdl_params))
        return mdl

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", lr_min, lr_max, log=True
            ),
        }

    # Reference computes max_steps explicitly for packed/iterable training.
    # This keeps walltime predictable and avoids relying on "epoch" semantics
    # when using an IterableDataset that repeats admissions `n_epochs` times.
    world_size = max(1, t.cuda.device_count())
    max_steps = max(
        1,
        dataset.n_train
        * n_epochs
        // per_device_train_batch_size
        // world_size
        // gradient_accumulation_steps,
    )
    resolved_eval_steps = int(eval_steps) if eval_steps is not None else max(
        1, max_steps // max(1, n_epochs * evaluations_per_epoch)
    )

    pad_token_id = int(dataset.vocab("PAD"))
    processing_stub = _PretokenizedProcessingStub(
        pad_token_id=pad_token_id,
        model_max_length=max_seq_length,
    )

    def _format_pretokenized(example):
        return example["input_ids"]

    def _collate_pretokenized_batch(
        examples: list[dict[str, typing.Any]],
    ) -> dict[str, t.Tensor]:
        batch_size = len(examples)
        max_len = max(int(t.as_tensor(ex["input_ids"]).numel()) for ex in examples)
        input_ids = t.full((batch_size, max_len), fill_value=pad_token_id, dtype=t.long)
        for i, ex in enumerate(examples):
            ids = t.as_tensor(ex["input_ids"], dtype=t.long).reshape(-1)
            input_ids[i, : ids.numel()] = ids
        attention_mask = (input_ids != pad_token_id).long()
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Train directly from pretokenized packed sequences. This avoids SFTTrainer
    # formatting/tokenization paths and exposes the same optimizer/checkpoint
    # contract used by Exp2/3.
    use_bf16 = bool(use_bf16) and t.cuda.is_available()
    training_args = TrainingArguments(
        report_to="wandb",
        run_name="{m}-{j}".format(m=model_version, j=jid),
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=float(adam_beta1),
        adam_beta2=float(adam_beta2),
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        tf32=True,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        include_num_input_tokens_seen="all",
        num_train_epochs=n_epochs,
        save_total_limit=save_total_limit,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        eval_strategy="steps",
        eval_steps=resolved_eval_steps,
        save_strategy="steps",
        save_steps=resolved_eval_steps,
        max_steps=max_steps,
        ddp_find_unused_parameters=False,
        seed=seed,
        data_seed=seed,
    )

    trainer = PackedIRBTrainer(
        model=model_init(),
        model_init=model_init,
        train_dataset=dataset.get_train_dataset(
            n_epochs=n_epochs, iterable=iterable_dataset
        ),
        eval_dataset=dataset.get_val_dataset(iterable=iterable_dataset),
        args=training_args,
        data_collator=_collate_pretokenized_batch,
        callbacks=[
            # Shared across Exp1-Exp3 via IRB_STAGE1_EARLY_STOPPING_PATIENCE.
            *(
                [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
                if early_stopping_patience > 0
                else []
            ),
            NanStoppingCallback(),
            LiveProgressCallback(
                output_dir,
                interval_seconds=progress_interval_seconds,
            ),
            PeriodicCheckpointCallback(checkpoint_interval_seconds),
            SignalCheckpointCallback(),
            CheckpointCompletionCallback(),
        ],
    )

    best_mdl_loc = model_dir.joinpath(
        "{m}-{j}-hp-{d}".format(m=model_version, j=jid, d=data_version)
    )
    # If we already exported a "best model" pointer, do nothing (all ranks exit cleanly).
    if best_mdl_loc.exists() or best_mdl_loc.is_symlink():
        return str(best_mdl_loc) if trainer.is_world_process_zero() else None

    best_ckpt = None
    if do_hpo:
        if resume_from_checkpoint:
            logger.warning(
                "resume_from_checkpoint is ignored when do_hpo=True (current HPO path does not resume trials)."
            )
        best_trial = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=n_trials,
        )
        if trainer.is_world_process_zero():
            best_ckpt = sorted(
                output_dir.joinpath(f"run-{best_trial.run_id}").glob("checkpoint-*")
            ).pop()
    else:
        # Fixed-hyperparameter training
        trainer.args.learning_rate = learning_rate
        trainer.args.gradient_accumulation_steps = gradient_accumulation_steps
        resume_ckpt = (
            str(pathlib.Path(resume_from_checkpoint).expanduser().resolve())
            if resume_from_checkpoint
            else None
        )
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()
            t.cuda.empty_cache()
        trainer.train(resume_from_checkpoint=resume_ckpt)

        if trainer.is_world_process_zero():
            best_ckpt = (
                pathlib.Path(trainer.state.best_model_checkpoint)
                if trainer.state.best_model_checkpoint is not None
                else (
                    pathlib.Path(_latest_complete_checkpoint(output_dir))
                    if _latest_complete_checkpoint(output_dir) is not None
                    else _resolve_latest_checkpoint(output_dir)
                )
            )
            if best_ckpt is None:
                raise RuntimeError(
                    "No best checkpoint found after training under "
                    f"{output_dir} (checked checkpoint-* and run-*/checkpoint-*)."
                )

    if trainer.is_world_process_zero():
        if best_ckpt is None:
            raise RuntimeError("best_ckpt was not resolved on rank0.")
        _write_loss_history(output_dir, list(trainer.state.log_history))
        grad_norms = [
            float(entry["grad_norm"])
            for entry in trainer.state.log_history
            if entry.get("grad_norm") is not None
        ]
        run_record = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": training_args.run_name,
            "model_name": model_name,
            "model_version": model_version,
            "rope_theta": rope_theta,
            "data_version": data_version,
            "seed": training_args.seed,
            "optimizer": {
                "name": "adamw",
                "learning_rate": float(learning_rate),
                "adam_beta1": float(adam_beta1),
                "adam_beta2": float(adam_beta2),
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "lr_scheduler_type": str(training_args.lr_scheduler_type),
                "warmup_ratio": warmup_ratio,
            },
            "checkpoint_selection": {
                "best_model_checkpoint": str(best_ckpt),
                "best_metric": trainer.state.best_metric,
                "load_best_model_at_end": training_args.load_best_model_at_end,
                "eval_steps": resolved_eval_steps,
                "save_steps": resolved_eval_steps,
            },
            "training": {
                "global_step": trainer.state.global_step,
                "epoch": trainer.state.epoch,
                "collation": collation,
                "max_steps": int(max_steps),
                "evaluations_per_epoch": int(evaluations_per_epoch),
                "per_device_train_batch_size": int(
                    training_args.per_device_train_batch_size
                ),
                "per_device_eval_batch_size": int(
                    training_args.per_device_eval_batch_size
                ),
                "gradient_accumulation_steps": int(
                    training_args.gradient_accumulation_steps
                ),
                "effective_batch_sequences": int(
                    training_args.per_device_train_batch_size
                    * training_args.gradient_accumulation_steps
                    * max(1, int(os.getenv("WORLD_SIZE", "1")))
                ),
                "max_sequence_length": int(max_seq_length),
                "max_input_tokens_seen": max(
                    (
                        int(entry["num_input_tokens_seen"])
                        for entry in trainer.state.log_history
                        if entry.get("num_input_tokens_seen") is not None
                    ),
                    default=None,
                ),
                "gpu_max_memory_allocated_bytes": (
                    int(t.cuda.max_memory_allocated()) if t.cuda.is_available() else None
                ),
                "gpu_max_memory_reserved_bytes": (
                    int(t.cuda.max_memory_reserved()) if t.cuda.is_available() else None
                ),
                "parameter_count": int(
                    sum(p.numel() for p in trainer.model.parameters())
                ),
                "gradient_norm_observations": len(grad_norms),
                "gradient_norm_clipped_observations": sum(
                    value > max_grad_norm for value in grad_norms
                ),
                "gradient_norm_clipped_fraction": (
                    sum(value > max_grad_norm for value in grad_norms) / len(grad_norms)
                    if grad_norms
                    else None
                ),
                "arguments": training_args.to_dict(),
            },
        }
        (output_dir / "run_record.json").write_text(
            json.dumps(run_record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        best_mdl_loc.parent.mkdir(parents=True, exist_ok=True)
        best_mdl_loc.symlink_to(best_ckpt, target_is_directory=True)

        # Ensure group perms on the symlink target directory are already correct (created by Trainer);
        # still set perms on the parent to avoid surprises in shared group contexts.
        try:
            os.chown(best_mdl_loc, uid=-1, gid=os.stat(best_mdl_loc.parent).st_gid)
        except Exception:
            pass

        _log_wandb_directory_artifact(
            directory=best_ckpt,
            artifact_name=f"{model_version}-{jid}-best-checkpoint",
            artifact_type="model-checkpoint",
            metadata={
                "model_version": model_version,
                "rope_theta": rope_theta,
                "data_version": data_version,
                "slurm_jid": jid,
                "do_hpo": bool(do_hpo),
                "n_trials": int(n_trials),
                "learning_rate": float(learning_rate),
                "best_checkpoint_path": str(best_ckpt),
                "best_model_symlink": str(best_mdl_loc),
            },
            project=wandb_project,
            run_name=f"{model_version}-{jid}",
            require_wandb=require_wandb,
        )

        return str(best_mdl_loc)

    return None


if __name__ == "__main__":
    fi.Fire(main)
