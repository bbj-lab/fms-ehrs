#!/usr/bin/env python3

"""
grab all hidden states from each provided sequence;
Cf. extract_hidden_states
"""

import argparse
import pathlib

import numpy as np
import torch as t
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
parser.add_argument("--data_dir", type=pathlib.Path, default=None)
parser.add_argument("--data_version", type=str, default="day_stays_first_24h")
parser.add_argument("--model_loc", type=pathlib.Path, default=None)
parser.add_argument("--small_batch_sz", type=int, default=2**4)
parser.add_argument("--big_batch_sz", type=int, default=2**12)
parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
parser.add_argument("--out_dir", type=pathlib.Path, default=None)
parser.add_argument("--all_layers", action="store_true")
parser.add_argument("--batch_num_start", type=int, default=None)
parser.add_argument("--batch_num_end", type=int, default=None)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

out_dir = args.out_dir if args.out_dir else args.data_dir

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, out_dir),
)

t.cuda.set_device(device := t.device("cuda:0"))

# load and prep data
all_splits = ("train", "val", "test")
data_dirs = dict()
out_dirs = dict()
for s in all_splits:
    data_dirs[s] = data_dir / f"{args.data_version}-tokenized" / s
    out_dirs[s] = out_dir / f"{args.data_version}-tokenized" / s
    out_dirs[s].mkdir(exist_ok=True, parents=True)

vocab = Vocabulary().load(data_dirs["train"] / "vocab.gzip")

dataset = (
    load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s] / "tokens_timelines.parquet") for s in args.splits
        },
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

# iterate over splits and run inference using model
stop_tokens = t.tensor([vocab("PAD")]).to(device)

for s in args.splits:
    n = dataset[s].num_rows
    tl_len = len(dataset[s].select(range(1))["input_ids"][0])
    for batch_num, big_batch in tqdm(
        enumerate(t.split(t.arange(n), args.big_batch_sz))
    ):
        if args.batch_num_start is not None and batch_num < args.batch_num_start:
            continue
        if args.batch_num_end is not None and batch_num >= args.batch_num_end:
            continue
        features = (
            np.empty((big_batch.size(0), tl_len, d, h + 1), dtype=np.float16)
            if args.all_layers
            else np.empty((big_batch.size(0), tl_len, d), dtype=np.float16)
        )
        for small_batch in t.split(big_batch, args.small_batch_sz):
            batch = dataset[s]["input_ids"][small_batch]
            with t.inference_mode(), t.amp.autocast("cuda", dtype=t.float16):
                x = model.forward(input_ids=batch.to(device), output_hidden_states=True)
            feats = (
                t.stack(tuple(hs.detach().to("cpu") for hs in x.hidden_states), dim=-1)
                .numpy()
                .astype(np.float16)
                if args.all_layers
                else x.hidden_states[-1].detach().to("cpu").numpy().astype(np.float16)
            )
            first_stop_idx = t.argmax(
                t.isin(batch, stop_tokens).int(), dim=1, keepdim=True
            )  # or 0 if no stop token
            for i, j in enumerate(first_stop_idx.cpu().numpy().ravel()):
                if j > 0:
                    feats[i, j:] = np.nan
            features[small_batch - batch_num * args.big_batch_sz] = feats
            # t.cuda.empty_cache()
        set_perms(np.save, compress=True)(
            out_dirs[s]
            / "all-features{x}-{m}-batch{n}.npy".format(
                x="-all-layers" if args.all_layers else "",
                m=model_loc.stem,
                n=batch_num,
            ),
            features,
        )


logger.info("---fin")
