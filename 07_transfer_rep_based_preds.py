#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
break down performance by ICU admission type
"""

import collections
import os
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

from logger import get_logger
from util import set_pd_options, log_classification_metrics

set_pd_options()

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

k0 = set(locals().keys())
data_dir_orig: os.PathLike = "../clif-data"
data_dir_new: os.PathLike = "/scratch/burkh4rt/clif-data"
data_version: str = "day_stays_qc_first_24h"
model_loc: os.PathLike = "../clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630"
fast: bool = False
for k in sorted(locals().keys() - k0 - {"k0"}):
    logger.info(f"{k}: {locals()[k]}")

data_dir_orig, data_dir_new, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (data_dir_orig, data_dir_new, model_loc),
)

splits = ("train", "val", "test")
versions = ("orig", "new")
data_dirs = collections.defaultdict(dict)
for v in versions:
    for s in splits:
        data_dirs[v][s] = (data_dir_orig if v == "orig" else data_dir_new).joinpath(
            f"{data_version}-tokenized", s
        )

""" classification outcomes
"""

preds = collections.defaultdict(dict)
true = collections.defaultdict(dict)

for outcome in ("same_admission_death", "long_length_of_stay"):
    logger.info(outcome.replace("_", " ").upper().ljust(79, "-"))
    data = dict()
    for s in ("train", "val"):
        feats = np.load(
            data_dirs["orig"][s].joinpath("features-{m}.npy".format(m=model_loc.stem))
        )
        label = (
            pl.scan_parquet(
                data_dirs["orig"][s].joinpath("tokens_timelines_outcomes.parquet")
            )
            .select(outcome)
            .collect()
            .to_numpy()
            .ravel()
        )
        data[s] = lgb.Dataset(feats, label=label)
    bst = lgb.train(
        {"metric": "auc", "objective": "binary", "force_col_wise": True}
        | ({} if fast else {"learning_rate": 0.05}),
        data["train"],
        10 if fast else 1000,
        valid_sets=[data["val"]],
    )
    for v in versions:
        logger.info(v.upper())
        preds[outcome][v] = bst.predict(
            np.load(
                data_dirs[v]["test"].joinpath(
                    "features-{m}.npy".format(m=model_loc.stem)
                )
            )
        )
        true[outcome][v] = (
            pl.scan_parquet(
                data_dirs[v]["test"].joinpath("tokens_timelines_outcomes.parquet")
            )
            .select(outcome)
            .cast(pl.Int64)
            .collect()
            .to_numpy()
        )
        log_classification_metrics(
            y_true=true[outcome][v], y_score=preds[outcome][v], logger=logger
        )

for v in versions:
    np.save(
        data_dirs[v]["test"].joinpath(
            "feat-based-mort-llos-preds-{m}.npy".format(m=model_loc.stem)
        ),
        np.column_stack(
            [preds["same_admission_death"][v], preds["long_length_of_stay"][v]]
        ),
    )
