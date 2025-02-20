#!/usr/bin/env python3

"""
train a small version of Mamba with a packing strategy and 
Poisson-distributed padding
"""

import itertools
import os
import pathlib

data_version = "day_stays_qc"
model_name = "state-spaces/mamba-130m-hf"
model_version = "small-packed-rev"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser().absolute()
jid = os.getenv("SLURM_JOB_ID", "")

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_PROJECT"] = "mamba_clif_mimic_packing"

import torch as t
from datasets import Features, IterableDataset, Sequence, Value, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from logger import get_logger
from vocabulary import Vocabulary

n_epochs = 10
max_seq_length = 1024
learning_rate = 2e-4
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
rng = t.Generator().manual_seed(42)


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

# locate data and vocab
splits = ("train", "val")
data_dirs = {
    s: hm.joinpath("clif-data", f"{data_version}-tokenized", s) for s in splits
}
vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
output_dir = hm.joinpath("clif-mdls", model_version)
output_dir.mkdir(exist_ok=True, parents=True)


def get_padding(
    tk: int = vocab("PAD"), poisson_rate: float = 7.0, generator: t.Generator = rng
):
    size = t.poisson(t.tensor(poisson_rate), generator=generator).to(t.uint8)
    return t.full(size=(size.item(),), fill_value=tk, dtype=t.uint8)


# chunk out the dataset in a way that teaches padding
def chunk_iterable(it, chunk_size: int = max_seq_length):
    ret: t.Tensor = t.Tensor(size=(0,))
    for eg in it:
        x = t.concat((eg["input_ids"], get_padding()))
        while x.size(dim=0) > 0:
            ndiff = min(chunk_size - ret.size(dim=0), x.size(dim=0))
            ret = t.concat((ret, x[:ndiff]))
            x = x[ndiff:]
            if ret.size(dim=0) == chunk_size:
                yield {"input_ids": ret.to(t.uint8)}
                ret = t.Tensor(size=(0,))


# load data
dataset = (
    load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines.parquet")) for s in splits
        },
    )
    .map(
        lambda batch: {"input_ids": batch["tokens"]},
        batched=True,
        remove_columns=["hospitalization_id", "tokens", "times", "seq_len", "padded"],
        features=Features({"input_ids": Sequence(Value("uint8"))}),
    )
    .with_format("torch")
)

train_set = IterableDataset.from_generator(
    lambda: chunk_iterable(
        IterableDataset.from_generator(
            lambda: itertools.chain.from_iterable(
                itertools.repeat(iter(dataset["train"]), n_epochs)
            )
        ).shuffle(seed=42, buffer_size=1024)
    ),
    features=Features({"input_ids": Sequence(Value("uint8"))}),
)

val_set = IterableDataset.from_generator(
    lambda: chunk_iterable(dataset["val"]),
    features=Features({"input_ids": Sequence(Value("uint8"))}),
)

# grab a small mamba for training
config = AutoConfig.from_pretrained(
    model_name,
    vocab_size=len(vocab),
    bos_token_id=vocab("TL_START"),
    eos_token_id=vocab("TL_END"),
    pad_token_id=vocab("PAD"),
    # hidden_size=2**6,  # 768 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    # n_layer=2**4,  # 24 -- ibid
    # num_hidden_layers=2**4,  # 24 -- ibid
    # state_size=2**3,  # 16 -- ibid
)
model = AutoModelForCausalLM.from_config(config)

max_steps = (
    dataset["train"].num_rows
    * n_epochs
    // per_device_train_batch_size
    // t.cuda.device_count()
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
    load_best_model_at_end=True,
    # neftune_noise_alpha=5,
    eval_strategy="steps",
    save_strategy="steps",
    max_steps=max_steps,
    max_seq_length=max_seq_length,
)

trainer = SFTTrainer(
    model,
    train_dataset=train_set,
    eval_dataset=val_set,
    args=training_args,
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
