#!/usr/bin/env python3

"""
grab the final hidden state from each provided sequence
"""

import pathlib

from transformers import AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

import numpy as np
import torch as t
import torch.distributed as dist

from vocabulary import Vocabulary

data_version = "day-stays"
model_version = "small"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

# prepare parallelism
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = t.device(f"cuda:{rank}")
t.cuda.set_device(device)

# load and prep data
splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath("clif-data", f"{data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
output_dir = hm.joinpath("clif-mdls", model_version)

dataset = load_dataset(
    "parquet",
    data_files={
        s: str(data_dirs[s].joinpath("tokens_timelines.parquet")) for s in splits
    },
).map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
dataset.set_format("torch")

# load and prep model
model = AutoModelForCausalLM.from_pretrained(
    output_dir.joinpath("mdl-day-stays-small-2025-01-29T05:56:46-06:00")
    # output_dir.joinpath("mdl-day-stays-smallest-2025-01-28T22:16:07-06:00")
).to(device)
d = model.config.hidden_size
model = model.to(device)
model = t.nn.parallel.DistributedDataParallel(model, device_ids=[rank])


# iterate over splits and run inference using model
batch_sz = 2**7
features = dict()

for s in splits:
    if rank == 0:
        print(s)
    n = dataset[s].num_rows
    features[s] = np.empty((n, d))
    for batch_idx in tqdm(t.split(t.arange(n), batch_sz)):
        batch = dataset[s]["input_ids"][batch_idx].to(device)
        with t.no_grad():
            x = model.forward(input_ids=batch, output_hidden_states=True)
            features[s][batch_idx] = x.hidden_states[-1][:, -1, :].detach().to("cpu")

# save out results
for s in splits:
    np.savez(data_dirs[s].joinpath("features__.npy"), features[s])
