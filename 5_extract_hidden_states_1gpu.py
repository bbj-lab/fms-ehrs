import pathlib

from transformers import AutoModelForCausalLM
from datasets import load_dataset

import torch as t

from tqdm import tqdm

from vocabulary import Vocabulary

data_version = "day-stays"
model_version = "small"  # "smallest"

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath("clif-data", f"{data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
output_dir = hm.joinpath("clif-mdls", model_version)

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
t.set_default_device("cpu")

model = AutoModelForCausalLM.from_pretrained(
    output_dir.joinpath("mdl-day-stays-small-2025-01-29T05:56:46-06:00")
    # output_dir.joinpath("mdl-day-stays-smallest-2025-01-28T22:16:07-06:00")
).to(device)

dataset = load_dataset(
    "parquet",
    data_files={
        s: str(data_dirs[s].joinpath("tokens_timelines.parquet")) for s in splits
    },
).map(lambda batch: {"input_ids": batch["padded"]}, batched=True)

dataset.set_format("torch")

batch_sz = 32
d = model.config.hidden_size
features = dict()
s = "train"
n = dataset[s].num_rows
features[s] = t.empty(n, d)
for s in splits:
    print(s)
    for batch_idx in tqdm(t.split(t.arange(n), batch_sz)):
        batch = dataset[s]["input_ids"][batch_idx].to(device)
        with t.no_grad():
            x = model.forward(input_ids=batch, output_hidden_states=True)
            features[s][batch_idx] = x.hidden_states[-1][:, -1, :].to("cpu")

# %%
# save out results
import numpy as np

for s in splits:
    np.savez(data_dirs[s].joinpath("features_.npy"), features[s].detach())
