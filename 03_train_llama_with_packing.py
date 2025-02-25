#!/usr/bin/env python3

"""
train a small version of Mamba with a packing strategy and 
Poisson-distributed padding
"""

import os
import pathlib

data_version = "day_stays_qc"
model_version = "llama1b"
model_name = "meta-llama/Llama-3.2-1B"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser().absolute()
jid = os.getenv("SLURM_JOB_ID", "")

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_PROJECT"] = "mamba_clif_mimic_packing"
os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=model_version, j=jid)

import torch as t
from transformers import AutoConfig, AutoModelForCausalLM, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from logger import get_logger
from dataset import Datasets

n_epochs = 10
max_seq_length = 1024
learning_rate = 2e-4
per_device_train_batch_size = 4
per_device_eval_batch_size = 4

if os.getenv("RANK", "0") == "0":
    logger = get_logger()
    logger.info("running {}".format(__file__))
    logger.log_env()
    logger.info(f"{data_version=}")
    logger.info(f"{model_name=}")
    logger.info(f"{model_version=}")
    logger.info(f"{n_epochs=}")
    logger.info(f"{max_seq_length=}")
    logger.info(f"{learning_rate=}")
    logger.info(f"{per_device_train_batch_size=}")
    logger.info(f"{per_device_eval_batch_size=}")


output_dir = hm.joinpath("clif-mdls", "{m}-{j}".format(m=model_version, j=jid))
output_dir.mkdir(exist_ok=True, parents=True)

dataset = Datasets(data_version=data_version, hm=hm, collation="packed")


# grab a small mamba for training
config = AutoConfig.from_pretrained(
    model_name,
    vocab_size=len(dataset.vocab),
    bos_token_id=dataset.vocab("TL_START"),
    eos_token_id=dataset.vocab("TL_END"),
    pad_token_id=dataset.vocab("PAD"),
)
model = AutoModelForCausalLM.from_config(config)

max_steps = (
    dataset.n_train * n_epochs // per_device_train_batch_size // t.cuda.device_count()
)

# train model
training_args = SFTConfig(
    report_to="wandb",
    run_name="{m}-{j}".format(m=model_version, j=jid),
    output_dir=str(output_dir),
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=1,  # simulate larger batch sizes
    learning_rate=learning_rate,  # 2e-4 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    num_train_epochs=1,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    eval_strategy="steps",
    save_strategy="best",
    max_steps=max_steps,
    max_seq_length=max_seq_length,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset.get_train_dataset(n_epochs=n_epochs),
    eval_dataset=dataset.get_val_dataset(),
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()
trainer.save_model(
    str(
        output_dir.joinpath(
            "mdl-{d}-{m}-{j}".format(
                d=data_version,
                m=model_version,
                j=jid,
            )
        )
    )
)
