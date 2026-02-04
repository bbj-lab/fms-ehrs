#!/usr/bin/env python3

"""
grab the final hidden state from each provided sequence
"""

import argparse
import pathlib

import numpy as np
import torch as t
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="W++")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--batch_sz", type=int, default=2**5)
parser.add_argument("--all_layers", type=bool, action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")


data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

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
    data_dirs[s] = data_dir / f"{args.data_version}-tokenized" / s

vocab = Vocabulary().load(data_dirs["train"] / "vocab.gzip")

dataset = (
    load_dataset(
        "parquet",
        data_files={s: str(data_dirs[s] / "tokens_timelines.parquet") for s in splits},
    )
    .map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
    .with_format("torch")
)

# load and prep model
model = AutoModelForCausalLM.from_pretrained(
    model_loc, torch_dtype=t.float16
)  # in eval mode by default
d = model.config.hidden_size
h = model.config.num_hidden_layers
model = model.to(device)
if is_parallel:
    model = t.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

# iterate over splits and run inference using model
stop_tokens = t.tensor([vocab("PAD")]).to(device)

for s in splits:
    n = dataset[s].num_rows
    features = (
        np.empty((n, d, h + 1), dtype=np.float16)
        if args.all_layers
        else np.empty((n, d), dtype=np.float16)
    )
    for batch_idx in tqdm(t.split(t.arange(n), args.batch_sz)):
        batch = dataset[s]["input_ids"][batch_idx].to(device)
        final_nonpadding_idx = (
            t.argmax(t.isin(batch, stop_tokens).int(), dim=1, keepdim=True) - 1
        )
        with t.inference_mode():
            x = model.forward(input_ids=batch, output_hidden_states=True)
        ret = t.empty(
            size=(
                (final_nonpadding_idx.size(dim=0), d, h + 1)
                if args.all_layers
                else (final_nonpadding_idx.size(dim=0), d)
            ),
            dtype=x.hidden_states[-1].dtype,
            device=device,
        )
        x = t.stack(x.hidden_states, dim=-1) if args.all_layers else x.hidden_states[-1]
        for i, j in enumerate(final_nonpadding_idx):
            ret[i] = x[i, j]
        features[batch_idx] = ret.detach().to("cpu")
        t.cuda.empty_cache()

    set_perms(np.save)(
        data_dirs[s]
        / "features{x}-{m}.npy".format(
            x="-all-layers" if args.all_layers else "", m=model_loc.stem
        ),
        features,
    )  # save out result


logger.info("---fin")
