#!/usr/bin/env python3

"""
train a small version of Mamba on our tokenized & padded data
"""

import os
import pathlib

data_version = "day_stays_qc"
model_version = "medium"
model_name = (
    "state-spaces/mamba-130m-hf"
    if model_version.startswith("small")
    else "state-spaces/mamba-370m-hf"
)
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser().absolute()
jid = os.getenv("SLURM_JOB_ID", "")

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_PROJECT"] = "test"  # "mamba_clif_mimic_qc"
os.environ["WANDB_RUN_NAME"] = "{m}-{j}".format(m=model_version, j=jid)

from transformers import AutoConfig, AutoModelForCausalLM, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from logger import get_logger
from dataset import Datasets

if os.getenv("RANK", "0") == "0":
    logger = get_logger()
    logger.info("running {}".format(__file__))

output_dir = hm.joinpath("clif-mdls", "{m}-{j}".format(m=model_version, j=jid))
output_dir.mkdir(exist_ok=True, parents=True)

dataset = Datasets(data_version=data_version, hm=hm, collation="padded")

# grab a small mamba for training
config = AutoConfig.from_pretrained(
    model_name,
    vocab_size=len(dataset.vocab),
    bos_token_id=dataset.vocab("TL_START"),
    eos_token_id=[dataset.vocab("TL_END"), dataset.vocab("TRUNC")],
    pad_token_id=dataset.vocab("PAD"),
)
model = AutoModelForCausalLM.from_config(config)

# train model
training_args = SFTConfig(
    report_to="wandb",
    run_name="{m}-{j}".format(m=model_version, j=jid),
    max_seq_length=1024,
    output_dir=str(output_dir),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,  # simulate larger batch sizes
    learning_rate=2e-4,  # 2e-4 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    num_train_epochs=1,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    eval_strategy="steps",
    save_strategy="best",
    ddp_find_unused_parameters=False,
)
trainer = SFTTrainer(
    model,
    train_dataset=dataset.get_train_dataset(),
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
