#!/usr/bin/env python3

"""
grab the final hidden state (at just under 24h) from each provided sequence
"""

import pathlib

import numpy as np
import torch as t
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from vocabulary import Vocabulary

data_version = "day_stays_qc_first_24h"
model_version = "small"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

device = t.device(f"cuda:0")
t.cuda.set_device(device)

# load and prep data
splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath("clif-data", f"{data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
model_dir = hm.joinpath("clif-mdls", model_version)

s = "test"
dataset = load_dataset(
    "parquet",
    data_files={s: str(data_dirs[s].joinpath("tokens_timelines.parquet"))},
).map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
dataset.set_format("torch")

# load and prep model
model = AutoModelForCausalLM.from_pretrained(
    model_dir.joinpath("mdl-day_stays_qc-small-2025-02-05T19:20:52-06:00")
).to(device)
d = model.config.hidden_size

# iterate over splits, place spacing tokens in the front (as opposed to the back),
# and generate 100 more tokens for each test timeline with the model
batch_sz = 2**7
k = 1000
n = dataset[s].num_rows
next_k = np.empty((n, k))
for batch_idx in tqdm(t.split(t.arange(n), batch_sz)):
    batch_rt_pad = dataset[s]["input_ids"][batch_idx].to(device)
    batch_lf_pad = t.full_like(batch_rt_pad, vocab("PAD"))
    final_nonpadding_idx = t.argmax((batch_rt_pad == vocab("PAD")).int(), axis=1)
    for i, j in enumerate(final_nonpadding_idx):
        batch_lf_pad[i, -j.item() :] = batch_rt_pad[i, : j.item()]
    new = model.generate(batch_lf_pad, max_new_tokens=k, do_sample=True, top_p=0.95)
    next_k[batch_idx] = new[:, -k:].detach().to("cpu")

# save out results
np.save(data_dirs[s].joinpath(f"next_k{k}.npy"), next_k)
