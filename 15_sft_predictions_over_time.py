#!/usr/bin/env python3

"""
examine how probabilistic predictions of outcomes evolve as timelines progress
"""

import os
import pathlib
import pickle

import datasets as ds
import numpy as np
import torch as t
from transformers import AutoModelForSequenceClassification, Trainer

from logger import get_logger
from vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


model_dir: os.PathLike = "../clif-mdls-archive/mdl-llama1b-sft-57451707-clsfr"
data_dir: os.PathLike = "../clif-data/day_stays_qc_first_24h-tokenized"

model_dir, data_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (model_dir, data_dir),
)

# load and prep data
rng = np.random.default_rng(42)
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(s) for s in splits}

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
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
trainer = Trainer(model=model)


def process_idx(i: int):
    tl = dataset["test"][i]["input_ids"].reshape(-1)
    hz = wd if (wd := t.argmax((tl == tk).int()).item()) > 0 else tl.shape[0]
    seq = t.stack(
        [t.concat([t.full((tl.shape[0] - i,), tk), tl[:i]]) for i in range(hz + 1)]
    )
    logits = trainer.predict(
        ds.Dataset.from_dict({"input_ids": seq.tolist()})
    ).predictions
    mort_probs = t.nn.functional.softmax(t.tensor(logits), dim=-1).numpy()[:, 1]
    return mort_probs


mort_true = dataset["test"]["label"].numpy()
mort_idx = np.nonzero(mort_true.astype(int).ravel())[0]
live_idx = np.setdiff1d(np.arange(mort_true.shape[0]), mort_idx)

mort_samp = rng.choice(mort_idx, size=5, replace=False).tolist()
live_samp = rng.choice(live_idx, size=5, replace=False).tolist()

mort_preds = {i: process_idx(i) for i in mort_samp}
live_preds = {i: process_idx(i) for i in live_samp}

with open(
    data_dirs["test"].joinpath("sft_preds_tokenwise-" + model_dir.stem + "-lite.pkl"),
    "wb",
) as fp:
    pickle.dump({"mort_preds": mort_preds, "live_preds": live_preds}, fp)
