#!/usr/bin/env python3

"""
train a small version of Mamba on our tokenized & padded data
"""

import datetime
import os
import pathlib

data_version = "day_stays_qc"
model_version = "minimal-version"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser().absolute()

os.environ["HF_HOME"] = "/gpfs/data/bbj-lab/cache/huggingface/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_DIR"] = "/scratch/burkh4rt/"
os.environ["WANDB_PROJECT"] = "mamba_clif_mimic_no_trunc"
os.environ["WANDB_RUN_NAME"] = "{d}-{m}".format(d=data_version, m=model_version)

from datasets import Features, Sequence, Value, load_dataset

# from torch import concat as t_concat, full as t_full, stack as t_stack, where as t_where
from transformers import AutoConfig, AutoModelForCausalLM  # , DataCollatorWithPadding
from trl import SFTConfig, SFTTrainer

from vocabulary import Vocabulary

# locate data and vocab
splits = ("train", "val")
data_dirs = {
    s: hm.joinpath("clif-data", f"{data_version}-tokenized", s) for s in splits
}
vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
output_dir = hm.joinpath("clif-mdls", model_version)
output_dir.mkdir(exist_ok=True, parents=True)

# grab a small mamba for training
model_name = "state-spaces/mamba-130m-hf"
config = AutoConfig.from_pretrained(
    model_name,
    vocab_size=len(vocab),
    bos_token_id=vocab("TL_START"),
    eos_token_id=vocab("TL_END"),
    pad_token_id=vocab("PAD"),
    hidden_size=2**6,  # 768 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    n_layer=2**4,  # 24 -- ibid
    num_hidden_layers=2**4,  # 24 -- ibid
    state_size=2**3,  # 16 -- ibid
)
model = AutoModelForCausalLM.from_config(config)

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
    .shuffle(seed=42)
)

# train model
training_args = SFTConfig(
    report_to="wandb",
    run_name=model_version,
    output_dir=str(output_dir),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,  # simulate larger batch sizes
    learning_rate=1e-4,  # 2e-4 -- cf. https://arxiv.org/pdf/2412.16178 tbl. 6
    num_train_epochs=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    # neftune_noise_alpha=5,
    eval_strategy="steps",
    save_strategy="steps",
)


# class PreTokenizedDataCollator(DataCollatorWithPadding):
#     def __init__(self, padding_token: int = vocab("PAD"), padding_side: str = "right"):
#         super().__init__(tokenizer=None)
#         self.padding_token = padding_token
#         assert padding_side in ("left", "right")
#         self.padding_side = padding_side
#
#     def __call__(self, features: list[dict[str, any]]) -> dict[str, any]:
#         batch = {key: [f[key] for f in features] for key in features[0]}
#         input_ids = batch["input_ids"]
#         max_batch_len = max(len(x) for x in input_ids)
#         input_ids = [
#             t_concat(
#                 [x, t_full((max_batch_len - x.shape[0],), self.padding_token)]
#                 if self.padding_side == "right"
#                 else [t_full((max_batch_len - x.shape[0],), self.padding_token), x]
#             )
#             for x in input_ids
#         ]
#         batch["input_ids"] = t_stack(input_ids)
#         batch["labels"] = t_where(
#             batch["input_ids"] == self.padding_token, -100, batch["input_ids"]
#         )
#         return batch


trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    args=training_args,
    # data_collator=PreTokenizedDataCollator(),
)
trainer.train()
trainer.save_model(
    str(
        output_dir.joinpath(
            "mdl-{d}-{m}-{t}".format(
                d=data_version,
                m=model_version,
                t=datetime.datetime.now(datetime.timezone.utc)
                .replace(microsecond=0)
                .astimezone()
                .isoformat(),
            )
        )
    )
)
