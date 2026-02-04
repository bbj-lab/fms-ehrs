#!/usr/bin/env python3

"""
grab the final hidden state from each provided sequence
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
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="W++")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--batch_sz", type=int, default=2**5)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")


data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

t.cuda.set_device(device := t.device("cuda:0"))

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
        columns=["padded"],
    )
    .map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
    .with_format("torch")
)

# load and prep model
model = AutoModelForCausalLM.from_pretrained(
    model_loc, torch_dtype=t.float16
)  # in eval mode by default
d = model.config.hidden_size
model = model.to(device)

# iterate over splits and run inference using model
stop_tokens = t.tensor([vocab("PAD"), vocab("TRUNC")]).to(device)

for s in splits:
    n = dataset[s].num_rows
    features = np.empty((n, d), dtype=np.float16)
    for batch_idx in tqdm(t.split(t.arange(n), args.batch_sz)):
        batch = dataset[s]["input_ids"][batch_idx].to(device)
        seq_idx = t.argmax(t.isin(batch, stop_tokens).int(), dim=1) - 1
        with t.inference_mode():
            hidden = model(input_ids=batch, output_hidden_states=True).hidden_states[-1]
        features[batch_idx] = hidden[
            t.arange(len(batch_idx), device=device), seq_idx
        ].cpu()
        t.cuda.empty_cache()

    set_perms(np.save)(
        data_dirs[s] / "features-{m}.npy".format(m=model_loc.stem), features
    )  # save out result


logger.info("---fin")
