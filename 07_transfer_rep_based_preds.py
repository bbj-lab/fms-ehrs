#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
break down performance by ICU admission type
"""

import argparse
import collections
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

parser = argparse.ArgumentParser(
    description="Learn representation-based classifiers and apply them to a new dataset."
)
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../clif-data")
parser.add_argument(
    "--data_dir_new", type=pathlib.Path, default="/scratch/burkh4rt/clif-data"
)
parser.add_argument("--data_version", type=str, default="day_stays_qc_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630",
)
parser.add_argument("--fast", type=bool, default=False)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc),
)
data_version = args.data_version
fast = bool(args.fast)

splits = ("train", "val", "test")
versions = ("orig", "new")
data_dirs = collections.defaultdict(dict)
outliers = collections.defaultdict(dict)
for v in versions:
    for s in splits:
        data_dirs[v][s] = (data_dir_orig if v == "orig" else data_dir_new).joinpath(
            f"{data_version}-tokenized", s
        )
        outliers[v][s] = (
            np.load(
                data_dirs[v][s].joinpath(
                    "features-outliers-{m}.npy".format(m=model_loc.stem)
                )
            )  # "Returns -1 for outliers and 1 for inliers"
            == -1
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
        logger.info("overall performance".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=true[outcome][v], y_score=preds[outcome][v], logger=logger
        )
        logger.info("on outliers".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=true[outcome][v][outliers[v]["test"]],
            y_score=preds[outcome][v][outliers[v]["test"]],
            logger=logger,
        )
        logger.info("on inliers".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=true[outcome][v][~outliers[v]["test"]],
            y_score=preds[outcome][v][~outliers[v]["test"]],
            logger=logger,
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
