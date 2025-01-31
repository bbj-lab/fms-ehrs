#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
"""

import os
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl
import sklearn.metrics as skl_mets

data_version = "first-24h"

if os.uname().nodename.startswith("cri"):
    hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data")
else:
    # change following line to develop locally
    hm = pathlib.Path(__file__).parent.joinpath("results").absolute()

splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath(f"{data_version}-tokenized", s)

""" classifier for admission mortality
"""
data = dict()
for s in ("train", "val"):
    feats = np.load(data_dirs[s].joinpath("features.npy"))
    mort = (
        pl.scan_parquet(data_dirs[s].joinpath("outcomes.parquet"))
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .ravel()
    )
    data[s] = lgb.Dataset(feats, label=mort)


bst = lgb.train(
    {"metric": "auc", "objective": "binary", "learning_rate": 0.05},
    data["train"],
    1000,
    valid_sets=[data["val"]],
)
mort_pred = bst.predict(np.load(data_dirs["test"].joinpath("features.npy")))
mort_true = (
    pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
    .select("same_admission_death")
    .cast(pl.Int64)
    .collect()
    .to_numpy()
)

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

""" regression for length of stay (in hours)
"""

for s in ("train", "val"):
    feats = np.load(data_dirs[s].joinpath("features.npy"))
    mort = (
        pl.scan_parquet(data_dirs[s].joinpath("outcomes.parquet"))
        .select("length_of_stay")
        .collect()
        .to_numpy()
        .ravel()
    )
    data[s] = lgb.Dataset(feats, label=mort)


bst = lgb.train(
    {"objective": "regression", "learning_rate": 0.05},
    data["train"],
    1000,
    valid_sets=[data["val"]],
)
los_pred = bst.predict(np.load(data_dirs["test"].joinpath("features.npy")))
los_true = (
    pl.scan_parquet(data_dirs["test"].joinpath("outcomes.parquet"))
    .select("length_of_stay")
    .cast(pl.Int64)
    .collect()
    .to_numpy()
)

for met in (
    "root_mean_squared_error",
    "mean_absolute_error",
    "explained_variance_score",
):
    print(
        "{}: {:.3f}".format(
            met,
            getattr(skl_mets, met)(y_true=los_true, y_pred=los_pred),
        )
    )
