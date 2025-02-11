#!/usr/bin/env python3

"""
collect n_samp repetitions of vllm predictions and process them
"""

import pathlib
import pickle

import numpy as np
import polars as pl
import sklearn.metrics as skl_mets

from vocabulary import Vocabulary

data_version = "day_stays_qc_first_24h"
model_version = "small"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()
k = 10_000

# load and prep data
splits = ("train", "val", "test")
data_dirs = {
    s: hm.joinpath("clif-data", f"{data_version}-tokenized", s) for s in splits
}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

determined = list()
expired = list()

for fp in data_dirs["test"].glob(f"responses_k{k}_rep_*_of_*.pkl"):
    det = list()
    exp = list()
    with open(fp, "rb") as f:
        response_list = pickle.load(f)
        for x in response_list:
            det.append(len(x) < k)
            exp.append(vocab("expired") in x)
    determined.append(det)
    expired.append(exp)

determined, expired = map(np.array, (determined, expired))

print("Determined: {}%".format(100 * determined.mean().round(5)))

mort_pred = expired.mean(axis=0)

mort_true = (
    pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
    .select("same_admission_death")
    .cast(pl.Int64)
    .collect()
    .to_numpy()
)

print(
    "{d} deaths in population of {pop} ({pct:.2f}%)".format(
        d=mort_true.sum(),
        pop=mort_true.size,
        pct=mort_true.mean() * 100,
    ),
)

print("{pred} predicted deaths".format(pred=mort_pred.sum().astype(int)))

print(
    "roc_auc: {:.3f}".format(
        skl_mets.roc_auc_score(y_true=mort_true, y_score=mort_pred)
    )
)

for met in (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
):
    print(
        "{}: {:.3f}".format(
            met,
            getattr(skl_mets, f"{met}_score")(
                y_true=mort_true, y_pred=np.round(mort_pred)
            ),
        )
    )
