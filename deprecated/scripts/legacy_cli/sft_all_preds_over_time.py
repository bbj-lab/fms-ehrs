#!/usr/bin/env python3

"""
examine how probabilistic predictions of outcomes evolve as timelines progress
"""

import argparse
import pathlib

import datasets as ds
import numpy as np
import torch as t
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--sft_model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death",
)
parser.add_argument("--batch_sz", type=int, default=2**5)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

sft_model_loc, data_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.sft_model_loc, args.data_dir),
)

# load and prep data
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

dataset = (
    ds.load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
            for s in ("test",)
        },
        columns=["padded", "same_admission_death"],
    )
    .with_format("torch")
    .map(
        lambda x: {"input_ids": x["padded"], "label": x["same_admission_death"]},
        remove_columns=["padded", "same_admission_death"],
    )
)

device = t.device("cuda:0")
model = AutoModelForSequenceClassification.from_pretrained(sft_model_loc).to(device)
n = dataset["test"].num_rows
nt = dataset["test"]["input_ids"][0].shape[0]
ret = t.empty(size=(n, nt), dtype=t.float, device="cpu")

for batch_idx in tqdm(t.split(t.arange(n), args.batch_sz)):
    batch = dataset["test"]["input_ids"][batch_idx].to(device)
    for it in tqdm(range(nt), leave=False):
        with t.inference_mode():
            x = (
                model.forward(input_ids=batch[:, it].reshape(-1, 1))
                if it == 0
                else model.forward(
                    input_ids=batch[:, it].reshape(-1, 1),
                    past_key_values=x.past_key_values,  # noqa -- x will exist when called
                )
            )
            # prior to the first padding index, this works like:
            # x = model.forward(input_ids=batch[:, : it + 1].reshape(-1, it + 1))
            # but is much faster
        ret[batch_idx, it] = t.nn.functional.softmax(x.logits, dim=-1)[:, 1].cpu()

set_perms(np.save, compress=True)(
    data_dirs["test"].joinpath(
        "sft_all_preds_over_time_" + sft_model_loc.stem + ".npy.gz"
    ),
    ret,
)


logger.info("---fin")
