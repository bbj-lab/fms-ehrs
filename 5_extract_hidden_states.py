#!/usr/bin/env python3

"""
grab the final hidden state (at just under 24h) from each provided sequence
"""

import pathlib

import numpy as np
import torch as t
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from vocabulary import Vocabulary

data_version = "day_stays_qc_first_24h"
model_version = "small-lr-search"  # "small"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

# prepare parallelism
is_parallel = t.cuda.device_count() > 1
if is_parallel:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
else:
    rank = 0
device = t.device(f"cuda:{rank}")
t.cuda.set_device(device)

# load and prep data
splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath("clif-data", f"{data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
mdl_dir = hm.joinpath("clif-mdls", model_version)

dataset = (
    load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines.parquet")) for s in splits
        },
    )
    .map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
    .with_format("torch")
)

# load and prep model
model = AutoModelForCausalLM.from_pretrained(
    mdl_dir.joinpath("run-1", "checkpoint-9000")
)
d = model.config.hidden_size
model = model.to(device)
if is_parallel:
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
        final_nonpadding_idx = (
            t.argmax((batch == vocab("PAD")).int(), axis=1, keepdim=True) - 1
        )
        with t.no_grad():
            x = model.forward(input_ids=batch, output_hidden_states=True)
            ret = t.empty(
                size=(final_nonpadding_idx.size(dim=0), d),
                dtype=x.hidden_states[-1].dtype,
                device=device,
            )
            for i, j in enumerate(final_nonpadding_idx):
                ret[i] = x.hidden_states[-1][i, j, :]
            features[s][batch_idx] = ret.detach().to("cpu")

# save out results
for s in splits:
    np.save(
        data_dirs[s].joinpath("features-{m}.npy".format(m=model_version)), features[s]
    )
