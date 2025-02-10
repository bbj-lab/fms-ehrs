#!/usr/bin/env python3

"""
take generated timelines and see if either death or discharge is predicted;
when possible, predict mortality and compare with true outcomes
"""

import pathlib

import numpy as np
import polars as pl
import sklearn.metrics as skl_mets

from vocabulary import Vocabulary

data_version = "day_stays_qc_first_24h"
k = 1000
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

# load and prep data
splits = ("train", "val", "test")
data_dirs = {
    s: hm.joinpath("clif-data", f"{data_version}-tokenized", s) for s in splits
}
next_k = np.load(data_dirs["test"].joinpath(f"next_k{k}.npy"))

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

mort_flags = next_k == vocab("expired")
first_mort_token = np.where(
    np.any(mort_flags, axis=1),
    np.argmax(mort_flags, axis=1),
    np.inf,
)
dis_flags = np.isin(next_k, test_elements=[vocab("TL_END"), vocab("PAD")])
first_discharge_token = np.where(
    np.any(dis_flags, axis=1), np.argmax(dis_flags, axis=1), np.inf
)

valid_pred = np.logical_or(
    np.isfinite(first_mort_token), np.isfinite(first_discharge_token)
)
pred_life = np.logical_and(valid_pred, first_discharge_token < first_mort_token)
mort_pred = np.logical_and(valid_pred, first_discharge_token > first_mort_token)

assert np.array_equal(valid_pred, np.logical_or(pred_life, mort_pred))

mort_true = (
    pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
    .select("same_admission_death")
    .cast(pl.Int64)
    .collect()
    .to_numpy()
)

print(
    "Predictions made for {tot} patients ({pct:.2f}%)".format(
        tot=valid_pred.sum(), pct=valid_pred.mean() * 100
    )
)

print(
    "{d} deaths in population of {pop}".format(
        d=mort_true[valid_pred].sum(), pop=valid_pred.sum()
    )
)

print("{pred} predicted deaths".format(pred=mort_pred[valid_pred].sum()))

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
                y_true=mort_true[valid_pred], y_pred=mort_pred[valid_pred]
            ),
        )
    )
