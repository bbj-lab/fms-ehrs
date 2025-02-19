#!/usr/bin/env python3

"""
train a small version of Mamba with a packing strategy and
Poisson-distributed padding
"""

import itertools
import os
import pathlib

data_version = "day_stays_qc"
model_version = "small-packing-search"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser().absolute()

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_PROJECT"] = "mamba_clif_mimic_packing"
os.environ["WANDB_RUN_NAME"] = model_version

import torch as t
from datasets import Features, IterableDataset, Sequence, Value, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from vocabulary import Vocabulary

n_epochs = 5
max_seq_length = 1024
rng = t.Generator().manual_seed(42)

# locate data and vocab
splits = ("train", "val")
data_dirs = {
    s: hm.joinpath("clif-data", f"{data_version}-tokenized", s) for s in splits
}
vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
output_dir = hm.joinpath("clif-mdls", model_version)
output_dir.mkdir(exist_ok=True, parents=True)


def model_init(trial=None):
    # grab a small mamba for training
    model_name = "state-spaces/mamba-130m-hf"
    config = AutoConfig.from_pretrained(
        model_name,
        # hidden_size=2**6,  # 768 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
        # n_layer=2**4,  # 24 -- ibid
        # num_hidden_layers=2**4,  # 24 -- ibid
        # state_size=2**3,  # 16 -- ibid
        vocab_size=len(vocab),
        bos_token_id=vocab("TL_START"),
        eos_token_id=vocab("TL_END"),
        pad_token_id=vocab("PAD"),
    )
    return AutoModelForCausalLM.from_config(config)


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

train_me = IterableDataset.from_generator(
    lambda: chunk_iterable(
        IterableDataset.from_generator(
            lambda: itertools.chain.from_iterable(
                itertools.repeat(iter(dataset["train"]), n_epochs)
            )
        ).shuffle(seed=42, buffer_size=1024)
    ),
    features=Features({"input_ids": Sequence(Value("uint8"))}),
)

validate_me = IterableDataset.from_generator(
    lambda: chunk_iterable(dataset["val"]),
    features=Features({"input_ids": Sequence(Value("uint8"))}),
)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16]
        ),
        "gradient_accumulation_steps": trial.suggest_int(
            "gradient_accumulation_steps", 1, 3
        ),
    }


min_batch_size = 4
max_steps = (
    dataset["train"].num_rows * n_epochs // min_batch_size // t.cuda.device_count()
)


# train model
training_args = SFTConfig(
    report_to="wandb",
    run_name=model_version,
    max_seq_length=max_seq_length,
    output_dir=str(output_dir),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # simulate larger batch sizes
    learning_rate=2e-4,  # 2e-4 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    num_train_epochs=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    # neftune_noise_alpha=5,
    eval_strategy="steps",
    save_strategy="steps",
    max_steps=max_steps,
)

trainer = SFTTrainer(
    model=model_init(),
    model_init=model_init,
    train_dataset=train_me,
    eval_dataset=validate_me,
    args=training_args,
)

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=5,
)

print(best_trial)
