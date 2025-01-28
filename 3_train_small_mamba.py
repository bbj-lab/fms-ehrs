#!/usr/bin/env python3

"""
train a small version of Mamba on our tokenized & padded data
"""

import datetime
import os
import pathlib

data_version = "day-stays"
model_version = os.environ["WANDB_RUN_NAME"] = "neftune"

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_PROJECT"] = "clif_mamba"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from vocabulary import Vocabulary

# locate data and vocab
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()
splits = ("train", "val")
data_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath("clif-data", f"{data_version}-tokenized", s)
vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
output_dir = hm.joinpath("clif-mdls", model_version)
output_dir.mkdir(exist_ok=True, parents=True)

# grab a small mamba for training
model_name = "state-spaces/mamba-130m-hf"
config = AutoConfig.from_pretrained(
    model_name,
    vocab_size=len(vocab),
    bos_token_id=vocab("TL_START"),
    eos_token_id=[vocab("TL_END"), vocab("TRUNC")],
    pad_token_id=vocab("PAD"),
)
model = AutoModelForCausalLM.from_config(config)

# load data
dataset = (
    load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines.parquet"))
            for s in ("train", "val")
        },
    )
    .map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
    .shuffle(seed=42)
)

# train model
training_args = SFTConfig(
    report_to="wandb",
    run_name=model_version,
    max_seq_length=1024,
    output_dir=str(output_dir),
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    num_train_epochs=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    neftune_noise_alpha=5,
    eval_strategy="steps",
    save_strategy="steps",
)
trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    args=training_args,
)
trainer.train()
trainer.save_model(
    str(
        output_dir.joinpath(
            "mdl-{}".format(
                datetime.datetime.now(datetime.timezone.utc)
                .replace(microsecond=0)
                .astimezone()
                .isoformat()
            )
        )
    )
)
