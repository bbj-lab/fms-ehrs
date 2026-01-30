#!/usr/bin/env python3

"""
train a model with a packing strategy
"""

import os
import pathlib

import fire as fi
import numpy as np
import torch as t
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

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
    max_seq_length: int = 2048,
    data_version: str = "day_stays",
    model_version: str = "llama1b",
    model_name: str = "meta-llama/Llama-3.2-1B",
    per_device_train_batch_size: int = 4,
    lr: float = 2.5e-4,
    gr_acc: int = 4,
    data_dir: os.PathLike = None,
    model_dir: os.PathLike = None,
    jid: str = os.getenv("SLURM_JOB_ID", ""),
    wandb_project: str = None,
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
        data_version=data_version, data_dir=data_dir, max_seq_length=max_seq_length
    )

    conf_param = dict(
        vocab_size=len(dataset.vocab),
        bos_token_id=dataset.vocab("TL_START"),
        eos_token_id=dataset.vocab("TL_END"),
        pad_token_id=dataset.vocab("PAD"),
    )

    def model_init(trial=None):
        t.cuda.empty_cache()
        config = AutoConfig.from_pretrained(model_name, **conf_param, **kwargs)
        mdl = AutoModelForCausalLM.from_config(config)
        mdl_params = sum(p.numel() for p in mdl.parameters())
        logger.info("Model initialized, n. param = {}".format(mdl_params))
        return mdl

    # train model
    training_args = TrainingArguments(
        report_to="wandb",
        run_name="{m}-{j}".format(m=model_version, j=jid),
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=4,
        num_train_epochs=1,  # this is handled in our dataset object
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        eval_strategy="steps",
        save_strategy="best",
        ddp_find_unused_parameters=False,
        optim="paged_adamw_8bit",
        bf16=True,
    )

    def collate_fn(batch):
        input_ids = t.tensor([x["input_ids"] for x in batch])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    trainer = Trainer(
        model_init=model_init,
        data_collator=collate_fn,
        train_dataset=(tr_ds := dataset.get_train_dataset(n_epochs=n_epochs)),
        eval_dataset=dataset.get_val_dataset(),
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            NanStoppingCallback(),
        ],
    )

    logger.info(f"Training on {len(tr_ds)} sequences of length {max_seq_length}")

    trainer.train()
    best_ckpt = trainer.state.best_model_checkpoint
    set_perms(AutoModelForCausalLM.from_pretrained(best_ckpt).save_pretrained)(
        best_mdl_loc := model_dir
        / "{m}-{j}-hp-{d}".format(m=model_version, j=jid, d=data_version)
    )

    return best_mdl_loc


if __name__ == "__main__":
    fi.Fire(main)
