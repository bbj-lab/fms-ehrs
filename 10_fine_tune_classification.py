#!/usr/bin/env python3

"""
fine-tune a pretrained model for sequence classification
"""

import os
import pathlib

import datasets as ds
import numpy as np
import torch as t
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from logger import get_logger
from vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


model_dir = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").joinpath(
    "clif-mdls-archive",
    "mdl-day_stays_qc-llama1b-57350630",
)
data_dir = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").joinpath(
    "clif-data", "day_stays_qc_first_24h-tokenized"
)
out_dir = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").joinpath("clif-mdls")
model_version = "llama1b-sft"
n_epochs: int = 5
learning_rate: float = 2e-5
data_version: str = "day_stays_qc"
per_device_train_batch_size: int = 8
per_device_eval_batch_size: int = 8
gradient_accumulation_steps = 2
jid: str = os.getenv("SLURM_JOB_ID", "")
wandb_project: str = "mimic-sft-clsfr"

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_PROJECT"] = wandb_project
os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=model_version, j=jid)

output_dir = out_dir.joinpath("{m}-{j}".format(m=model_version, j=jid))
output_dir.mkdir(exist_ok=True, parents=True)

# load and prep data
splits = ("train", "val", "test")
np_rng = np.random.default_rng(42)
data_dirs = {s: data_dir.joinpath(s) for s in splits}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))


def rt_padding_to_left(t_rt):
    tk: int = vocab("PAD")
    i = t.argmax((t_rt == tk).int()).item()
    return t.concat([t.full((t_rt.shape[0] - i,), tk), t_rt[:i]])


dataset = (
    ds.load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
            for s in splits
        },
        columns=["padded", "same_admission_death"],
    )
    .with_format("torch")
    .map(
        lambda x: {
            "input_ids": rt_padding_to_left(x["padded"]),
            "label": x["same_admission_death"],
        },
        remove_columns=["padded", "same_admission_death"],
    )
)

model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# train model
training_args = TrainingArguments(
    report_to="wandb",
    run_name="{m}-{j}".format(m=model_version, j=jid),
    output_dir=str(output_dir),
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,  # simulate larger batch sizes
    learning_rate=learning_rate,  # 2e-4 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    num_train_epochs=n_epochs,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    eval_strategy="steps",
    save_strategy="best",
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model,
    train_dataset=dataset["train"].shuffle(generator=np_rng),
    eval_dataset=dataset["val"],
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()
trainer.save_model(
    str(
        output_dir.joinpath(
            "mdl-{m}-{j}-clsfr".format(
                m=model_version,
                j=jid,
            )
        )
    )
)
