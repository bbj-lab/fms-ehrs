#!/usr/bin/env python3

"""
tune a model with a packing strategy
"""

import os
import pathlib
import typing
import importlib.util

import fire as fi
import numpy as np
import torch as t
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
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
    learning_rate: float = 5e-5,
    lr_min: float = 5e-5,
    lr_max: float = 5e-4,
    # NOTE (benchmark fairness): gradient_accumulation_steps controls *data exposure*
    # (how many microbatches are consumed per optimizer step). We fix it to avoid
    # inadvertently giving some configurations more effective training tokens/compute.
    gradient_accumulation_steps: int = 2,
    # Optional performance knobs (objective-preserving):
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
        config = AutoConfig.from_pretrained(
            model_name,
            vocab_size=len(dataset.vocab),
            bos_token_id=dataset.vocab("TL_START"),
            eos_token_id=dataset.vocab("TL_END"),
            pad_token_id=dataset.vocab("PAD"),
            **cfg_kwargs,
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
    max_steps = (
            dataset.n_train
            * n_epochs
            // per_device_train_batch_size
            // t.cuda.device_count()
            // gradient_accumulation_steps
    )

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
        return str(best_mdl_loc) if os.getenv("RANK", "0") == "0" else None

    # IMPORTANT (DDP correctness):
    # All ranks must execute the same high-level control flow (HPO vs train),
    # otherwise DDP initialization / collectives will hang or error.
    best_ckpt = None
    if do_hpo:
        best_trial = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=n_trials,
        )
        # Each rank will compute `best_trial`, but only rank0 will export the checkpoint.
        if os.getenv("RANK", "0") == "0":
            best_ckpt = sorted(
                output_dir.joinpath(f"run-{best_trial.run_id}").glob("checkpoint-*")
            ).pop()
    else:
        # Fixed-hyperparameter training
        trainer.args.learning_rate = learning_rate
        trainer.args.gradient_accumulation_steps = gradient_accumulation_steps
        trainer.train()

        if os.getenv("RANK", "0") == "0":
            ckpts = sorted(output_dir.glob("checkpoint-*"))
            if not ckpts:
                raise RuntimeError(
                    f"No checkpoints found under {output_dir} after training."
                )
            best_ckpt = ckpts[-1]

    # Ensure all ranks finished (and fail fast consistently) before rank0 exports the symlink.
    if t.distributed.is_available() and t.distributed.is_initialized():
        # If any rank errors above, torchrun/elastic will tear down the job; barrier here
        # prevents silent rank-skew where rank0 exits early and others hang in collectives.
        t.distributed.barrier()

    if os.getenv("RANK", "0") == "0":
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

        return str(best_mdl_loc)

    return None


if __name__ == "__main__":
    fi.Fire(main)
