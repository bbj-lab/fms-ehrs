#!/usr/bin/env python3

"""
tune a model with a packing strategy
"""

import importlib.util
import json
import os
import pathlib
import typing

import fire as fi
import numpy as np
import torch as t
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    LlamaConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from fms_ehrs.framework.dataset import Datasets
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


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
        msg = (
            f"WANDB_MODE=offline; cannot upload artifact {artifact_name} from {directory}"
        )
        if require_wandb:
            raise RuntimeError(msg)
        logger.warning(msg)
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
    per_device_train_batch_size: int = 4,
    # max_grad_norm: float = 1.0,
    # Policy control:
    # - do_hpo=True: run Optuna HPO (expensive; may exceed cluster walltime)
    # - do_hpo=False: run a single fixed-hyperparameter training
    do_hpo: bool = True,
    # Optional resume path for continuing an interrupted fixed-hyperparameter run.
    resume_from_checkpoint: str | None = None,
    learning_rate: float = 5e-5,
    lr_min: float = 5e-5,
    lr_max: float = 5e-4,
    # NOTE (benchmark fairness): gradient_accumulation_steps controls *data exposure*
    # (how many microbatches are consumed per optimizer step). We fix it to avoid
    # inadvertently giving some configurations more effective training tokens/compute.
    gradient_accumulation_steps: int = 2,
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
    n_trials: int = 5,
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
    if isinstance(resume_from_checkpoint, str):
        s = resume_from_checkpoint.strip()
        if s.lower() in ("", "none", "null"):
            resume_from_checkpoint = None
        else:
            resume_from_checkpoint = s

    if int(gradient_accumulation_steps) != 2:
        raise ValueError(
            "For benchmark fairness, gradient_accumulation_steps is fixed to 2 "
            f"(got {gradient_accumulation_steps})."
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
        try:
            config = AutoConfig.from_pretrained(
                model_name,
                vocab_size=len(dataset.vocab),
                bos_token_id=dataset.vocab("TL_START"),
                eos_token_id=dataset.vocab("TL_END"),
                pad_token_id=dataset.vocab("PAD"),
                **cfg_kwargs,
            )
        except OSError as e:
            # Exp1 also trains from scratch, so only the base architecture config is needed.
            # Mirror Exp2/Exp3 behavior: if the upstream config repo is gated, fall back to
            # a local LlamaConfig with the same hyperparameters.
            msg = str(e).lower()
            is_gated = ("gated repo" in msg) or ("401 client error" in msg) or ("access to model" in msg)
            is_llama = "llama" in str(model_name).lower()
            if not (is_gated and is_llama):
                raise
            logger.warning(
                "AutoConfig.from_pretrained(%r) failed due to gated/unauthenticated access. "
                "Falling back to local LlamaConfig (random init; config-only). "
                "To use the upstream config, authenticate with HuggingFace and ensure you have access.",
                model_name,
            )
            config = LlamaConfig(
                vocab_size=len(dataset.vocab),
                bos_token_id=dataset.vocab("TL_START"),
                eos_token_id=dataset.vocab("TL_END"),
                pad_token_id=dataset.vocab("PAD"),
                **cfg_kwargs,
            )
        config._name_or_path = str(model_name)
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
    max_steps = (
            dataset.n_train
            * n_epochs
            // per_device_train_batch_size
            // t.cuda.device_count()
            // gradient_accumulation_steps
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

    # train model
    use_bf16 = bool(use_bf16) and t.cuda.is_available()
    training_args = SFTConfig(
        report_to="wandb",
        run_name="{m}-{j}".format(m=model_version, j=jid),
        max_seq_length=max_seq_length,
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        tf32=True,
        # max_grad_norm=max_grad_norm,
        num_train_epochs=1,  # this is handled in our dataset object
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        eval_strategy="steps",
        save_strategy="best",
        max_steps=max_steps,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model_init(),
        model_init=model_init,
        train_dataset=dataset.get_train_dataset(
            n_epochs=n_epochs, iterable=iterable_dataset
        ),
        eval_dataset=dataset.get_val_dataset(iterable=iterable_dataset),
        args=training_args,
        data_collator=_collate_pretokenized_batch,
        processing_class=processing_stub,
        formatting_func=_format_pretokenized,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            NanStoppingCallback(),
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
        trainer.train(resume_from_checkpoint=resume_ckpt)

        if trainer.is_world_process_zero():
            best_ckpt = _resolve_latest_checkpoint(output_dir)
            if best_ckpt is None:
                raise RuntimeError(
                    "No checkpoints found after training under "
                    f"{output_dir} (checked checkpoint-* and run-*/checkpoint-*)."
                )

    if trainer.is_world_process_zero():
        if best_ckpt is None:
            raise RuntimeError("best_ckpt was not resolved on rank0.")
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
