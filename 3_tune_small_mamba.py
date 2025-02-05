#!/usr/bin/env python3

"""
train a small version of Mamba on our tokenized & padded data
"""

import os
import pathlib

data_version = "day-stays"
model_version = "small-lr-search"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser().absolute()

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_DIR"] = hm.joinpath("wandb").__str__()
os.environ["WANDB_PROJECT"] = "mamba_clif_mimic"
os.environ["WANDB_RUN_NAME"] = model_version

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from vocabulary import Vocabulary

# locate data and vocab
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
    # hidden_size=25,  # 768 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    # n_layer=15,  # 24 -- ibid
    # num_hidden_layers=15,  # 24 -- ibid
    # state_size=16,  # 16 -- ibid
    vocab_size=len(vocab),
    bos_token_id=vocab("TL_START"),
    eos_token_id=[vocab("TL_END"), vocab("TRUNC")],
    pad_token_id=vocab("PAD"),
)


def model_init(trial):
    return AutoModelForCausalLM.from_config(config)


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


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32]
        ),
    }


# train model
training_args = SFTConfig(
    report_to="wandb",
    run_name=model_version,
    max_seq_length=1024,
    output_dir=str(output_dir),
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,  # 2e-4 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    num_train_epochs=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    # neftune_noise_alpha=5,
    eval_strategy="steps",
    save_strategy="steps",
)

trainer = SFTTrainer(
    model=model_init(None),
    model_init=model_init,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
    args=training_args,
)

best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
)

print(best_trial)

# trainer.train()
# trainer.save_model(
#     str(
#         output_dir.joinpath(
#             "mdl-{d}-{m}-{t}".format(
#                 d=data_version,
#                 m=model_version,
#                 t=datetime.datetime.now(datetime.timezone.utc)
#                 .replace(microsecond=0)
#                 .astimezone()
#                 .isoformat(),
#             )
#         )
#     )
# )
