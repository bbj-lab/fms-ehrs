#!/usr/bin/env python3

"""
examine how probabilistic predictions of outcomes evolve as timelines progress
"""

import argparse
import pathlib
import pickle

import datasets as ds
import numpy as np
import torch as t
from transformers import AutoModelForSequenceClassification, Trainer

from src.framework.logger import get_logger
from src.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path)
parser.add_argument("--data_version", type=str)
parser.add_argument("--model_loc", type=pathlib.Path)
parser.add_argument("--model_loc_urt", type=pathlib.Path)
parser.add_argument("--n", type=int, default=100)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

model_loc, model_loc_urt, data_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.model_loc, args.model_loc_urt, args.data_dir),
)
data_version = args.data_version

# load and prep data
rng = np.random.default_rng(42)
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{data_version}-tokenized", s) for s in splits}

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
        lambda x: {
            "input_ids": x["padded"],
            "label": x["same_admission_death"],
        },
        remove_columns=["padded", "same_admission_death"],
    )
)

device = t.device(f"cuda:0")

tk: int = vocab("PAD")

model = AutoModelForSequenceClassification.from_pretrained(model_loc)
trainer = Trainer(model=model)

model_urt = AutoModelForSequenceClassification.from_pretrained(model_loc_urt)
trainer_urt = Trainer(model=model_urt)


def process_idx(trainer: Trainer, i: int):
    tl = dataset["test"][i]["input_ids"].reshape(-1)
    hz = wd if (wd := t.argmax((tl == tk).int()).item()) > 0 else tl.shape[0]
    seq = t.stack(
        [t.concat([t.full((tl.shape[0] - i,), tk), tl[:i]]) for i in range(hz)]
    )
    logits = trainer.predict(
        ds.Dataset.from_dict({"input_ids": seq.tolist()})
    ).predictions
    mort_probs = t.nn.functional.softmax(t.tensor(logits), dim=-1).numpy()[:, 1]
    return mort_probs


mort_true = dataset["test"]["label"].numpy()
mort_idx = np.nonzero(mort_true.astype(int).ravel())[0]
live_idx = np.setdiff1d(np.arange(mort_true.shape[0]), mort_idx)

mort_samp = rng.choice(mort_idx, size=args.n, replace=False).tolist()
live_samp = rng.choice(live_idx, size=args.n, replace=False).tolist()

mort_preds = {i: process_idx(trainer, i) for i in mort_samp}
mort_preds_urt = {i: process_idx(trainer_urt, i) for i in mort_samp}
live_preds = {i: process_idx(trainer, i) for i in live_samp}
live_preds_urt = {i: process_idx(trainer_urt, i) for i in live_samp}

with open(
    data_dirs["test"].joinpath(
        "sft_preds-n" + str(args.n) + "_tokenwise-" + model_loc.stem + ".pkl"
    ),
    "wb",
) as fp:
    pickle.dump(
        {
            "mort_preds": mort_preds,
            "mort_preds_urt": mort_preds_urt,
            "live_preds": live_preds,
            "live_preds_urt": live_preds_urt,
        },
        fp,
    )
