#!/usr/bin/env python3

"""
tune a model with a packing strategy
"""

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
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from fms_ehrs.framework.dataset import Datasets
from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


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
    max_seq_length: int = 1024,
    data_version: str = "day_stays",
    model_version: str = "llama1b",
    model_name: str = "meta-llama/Llama-3.2-1B",
    per_device_train_batch_size: int = 4,
    max_grad_norm: float = 1.0,
    lr_min: float = 5e-5,
    lr_max: float = 5e-4,
    gr_acc_min: int = 1,
    gr_acc_max: int = 3,
    data_dir: os.PathLike = None,
    model_dir: os.PathLike = None,
    collation: typing.Literal["padded", "packed"] = "packed",
    jid: str = os.getenv("SLURM_JOB_ID", ""),
    wandb_project: str = None,
    n_trials: int = 5,
    iterable_dataset: bool = True,
    **kwargs,
):
    """pass additional model configuration parameters with kwargs"""

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
        config = AutoConfig.from_pretrained(
            model_name,
            vocab_size=len(dataset.vocab),
            bos_token_id=dataset.vocab("TL_START"),
            eos_token_id=dataset.vocab("TL_END"),
            pad_token_id=dataset.vocab("PAD"),
            **kwargs,
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
            "gradient_accumulation_steps": trial.suggest_int(
                "gradient_accumulation_steps", gr_acc_min, gr_acc_max
            ),
        }

    max_steps = (
        (
            dataset.n_train
            * n_epochs
            // per_device_train_batch_size
            // t.cuda.device_count()
        )
        if iterable_dataset
        else -1
    )

    # train model
    training_args = SFTConfig(
        report_to="wandb",
        run_name="{m}-{j}".format(m=model_version, j=jid),
        max_seq_length=max_seq_length,
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=4,
        max_grad_norm=max_grad_norm,
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

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=n_trials,
    )

    if os.getenv("RANK", "0") == "0":
        best_ckpt = sorted(
            output_dir.joinpath(f"run-{best_trial.run_id}").glob("checkpoint-*")
        ).pop()
        best_mdl_loc = model_dir.joinpath(
            "{m}-{j}-hp-{d}".format(m=model_version, j=jid, d=data_version)
        )
        set_perms(AutoModelForCausalLM.from_pretrained(best_ckpt).save_pretrained)(
            best_mdl_loc
        )
        return best_mdl_loc

    return None


if __name__ == "__main__":
    fi.Fire(main)


import pathlib
import polars as pl
from fms_ehrs.framework.vocabulary import Vocabulary

splits = ("train", "val", "test")
data_dir = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/data-mimic/W++-tokenized")
vocab = Vocabulary().load(data_dir.joinpath("train", "vocab.gzip"))
dfs = dict()
for s in splits:
    dfs[s] = pl.read_parquet(
        data_dir.joinpath(s, "tokens_timelines.parquet")
    ).with_columns(
        prefix=pl.col("tokens")
        .list.head(n=6)
        .list.eval(
            pl.element().map_elements(
                lambda v: vocab.reverse[v], return_dtype=pl.String
            )
        ),
        suffix=pl.col("tokens")
        .list.tail(2)
        .list.eval(
            pl.element().map_elements(
                lambda v: vocab.reverse[v], return_dtype=pl.String
            )
        ),
    )
