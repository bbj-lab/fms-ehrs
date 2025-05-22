#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import pathlib

import numpy as np
import torch as t
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from src.framework.logger import get_logger
from src.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama-med-58788824",
)
parser.add_argument("--batch_sz", type=int, default=2**8)
parser.add_argument("splits", nargs="*", default=["test"])
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc),
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
    data_dirs[s] = data_dir.joinpath(f"{args.data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

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
model = AutoModelForCausalLM.from_pretrained(model_loc)  # in eval mode by default
d = model.config.hidden_size
model = model.to(device)
if is_parallel:
    model = t.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

# iterate over splits and run inference using model
stop_tokens = t.tensor([vocab("PAD"), vocab("TRUNC"), vocab("TL_END")]).to(device)

for s in args.splits:
    n = dataset[s].num_rows
    tl_len = dataset[s].select(range(1))["input_ids"].shape[-1]
    log_probs = np.full(
        shape=(n, tl_len),
        fill_value=np.nan,
    )  # could use `np.empty` here, but perhaps safer this way

    for batch_idx in tqdm(t.split(t.arange(n), args.batch_sz)):
        batch = dataset[s]["input_ids"][batch_idx].to(device)
        with t.inference_mode():
            x = model.forward(input_ids=batch, output_hidden_states=True)
        log_probs_realized = (
            t.gather(
                t.nn.functional.log_softmax(
                    x.logits, dim=-1
                ),  # n_obs × tl_len × n_vocab
                -1,
                batch.unsqueeze(-1),
            )  # gather log prob for realized token
            .squeeze()
            .cpu()
            .numpy()
        )
        first_stop_idx = (
            t.argmax(t.isin(batch, stop_tokens).int(), axis=1, keepdim=True)
            .cpu()
            .numpy()
            .ravel()
        )
        for i, j in enumerate(first_stop_idx):
            if j > 0:
                log_probs_realized[i, j:] = np.nan
        log_probs[batch_idx] = log_probs_realized

    np.save(
        data_dirs[s].joinpath("log_probs-{m}.npy".format(m=model_loc.stem)),
        log_probs,
    )  # save out result

logger.info("---fin")
