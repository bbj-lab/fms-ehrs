#!/usr/bin/env python3

"""
tune a small model with a packing strategy
"""

import os
import pathlib
import typing

import fire as fi
import torch as t
from transformers import AutoConfig, AutoModelForCausalLM, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from fms_ehrs.framework.dataset import Datasets
from fms_ehrs.framework.logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    n_epochs: int = 5,
    max_seq_length: int = 1024,
    data_version: str = "day_stays",
    model_version: str = "llama1b",
    model_name: str = "meta-llama/Llama-3.2-1B",
    per_device_train_batch_size: int = 4,
    data_dir: os.PathLike = None,
    model_dir: os.PathLike = None,
    collation: typing.Literal["padded", "packed"] = "packed",
    jid: str = os.getenv("SLURM_JOB_ID", ""),
    wandb_project: str = None,
    n_trials: int = 5,
    **kwargs,
):
    """pass additional model configuration parameters with kwargs"""

    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=model_version, j=jid)

    data_dir, model_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (data_dir, model_dir),
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
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
            "gradient_accumulation_steps": trial.suggest_int(
                "gradient_accumulation_steps", 1, 3
            ),
        }

    max_steps = (
        dataset.n_train
        * n_epochs
        // per_device_train_batch_size
        // t.cuda.device_count()
    )

    # train model
    training_args = SFTConfig(
        report_to="wandb",
        run_name="{m}-{j}".format(m=model_version, j=jid),
        max_seq_length=max_seq_length,
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # simulate larger batch sizes
        learning_rate=2e-4,  # 2e-4 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
        num_train_epochs=1,
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
        train_dataset=dataset.get_train_dataset(n_epochs=n_epochs),
        eval_dataset=dataset.get_val_dataset(),
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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
        AutoModelForCausalLM.from_pretrained(best_ckpt).save_pretrained(best_mdl_loc)
        return best_mdl_loc

    return None


if __name__ == "__main__":
    fi.Fire(main)
