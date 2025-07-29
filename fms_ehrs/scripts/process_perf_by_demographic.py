#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
break down performance by ICU admission type
"""

import argparse
import functools
import pathlib
import pickle

import numpy as np
import pandas as pd
import polars as pl
import sklearn as skl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.util import set_pd_options
from fms_ehrs.framework.vocabulary import Vocabulary

set_pd_options()

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../data-ucmc")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../mdls-archive/llama-orig-58789721",
)
parser.add_argument(
    "--classifier",
    choices=["light_gbm", "logistic_regression_cv", "logistic_regression"],
    default="logistic_regression",
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc),
)

splits = ("train", "val", "test")
versions = ("orig", "new")
outcomes = ("same_admission_death", "long_length_of_stay", "imv_event", "icu_admission")

vocab = Vocabulary().load(
    data_dir_orig.joinpath(f"{args.data_version}-tokenized", "train", "vocab.gzip")
)
data_dirs = dict()
race = dict()
ethnicity = dict()
sex = dict()
preds = dict()

for v in versions:
    data_dirs[v] = (data_dir_orig if v == "orig" else data_dir_new).joinpath(
        f"{args.data_version}-tokenized", "test"
    )
    race[v] = (
        pl.scan_parquet(data_dirs[v].joinpath("tokens_timelines_outcomes.parquet"))
        .select(
            pl.col("tokens")
            .list.get(1)  # index of race token
            .map_elements(
                vocab.reverse.__getitem__,
                return_dtype=pl.String,
            )
            .replace(None, "Unknown/Other")
            .replace("other", "Unknown/Other")
            .replace("unknown", "Unknown/Other")
            .replace("american indian or alaska native", "Native American")
            .replace("asian", "Asian")
            .replace("black or african american", "African American")
            .replace("native hawaiian or other pacific islander", "Pacific Islander")
            .replace("white", "Caucasian")
        )
        .collect()
        .to_series()
        .to_numpy()
    )
    ethnicity[v] = (
        pl.scan_parquet(data_dirs[v].joinpath("tokens_timelines_outcomes.parquet"))
        .select(
            pl.col("tokens")
            .list.get(2)  # index of ethnicity token
            .map_elements(
                vocab.reverse.__getitem__,
                return_dtype=pl.String,
            )
            .replace(None, "unknown")
        )
        .collect()
        .to_series()
        .to_numpy()
    )
    sex[v] = (
        pl.scan_parquet(data_dirs[v].joinpath("tokens_timelines_outcomes.parquet"))
        .select(
            pl.col("tokens")
            .list.get(3)  # index of sex token
            .map_elements(
                vocab.reverse.__getitem__,
                return_dtype=pl.String,
            )
        )
        .collect()
        .to_series()
        .to_numpy()
    )
    with open(
        data_dirs[v].joinpath(args.classifier + "-preds-" + model_loc.stem + ".pkl"),
        "rb",
    ) as fp:
        preds[v] = pickle.load(fp)

races = sorted(functools.reduce(set.union, map(set, race.values())))
ethnicities = sorted(functools.reduce(set.union, map(set, ethnicity.values())))
sexes = sorted(functools.reduce(set.union, map(set, sex.values())))

for v in versions:
    print(v.upper().ljust(79, "-"))
    df = pd.DataFrame(index=races, columns=outcomes)
    for r in races:
        for out in outcomes:
            qualified_labels = preds[v]["labels"][out][
                np.logical_and(preds[v]["qualifiers"][out], race[v] == r)
            ]
            qualified_preds = preds[v]["predictions"][out][
                race[v][preds[v]["qualifiers"][out]] == r
            ]
            df.loc[r, out] = skl.metrics.roc_auc_score(
                y_true=qualified_labels, y_score=qualified_preds
            )
    print(df)
    df = pd.DataFrame(index=ethnicities, columns=outcomes)
    for r in ethnicities:
        for out in outcomes:
            qualified_labels = preds[v]["labels"][out][
                np.logical_and(preds[v]["qualifiers"][out], ethnicity[v] == r)
            ]
            qualified_preds = preds[v]["predictions"][out][
                ethnicity[v][preds[v]["qualifiers"][out]] == r
            ]
            df.loc[r, out] = skl.metrics.roc_auc_score(
                y_true=qualified_labels, y_score=qualified_preds
            )
    print(df)
    df = pd.DataFrame(index=sexes, columns=outcomes)
    for r in sexes:
        for out in outcomes:
            qualified_labels = preds[v]["labels"][out][
                np.logical_and(preds[v]["qualifiers"][out], sex[v] == r)
            ]
            qualified_preds = preds[v]["predictions"][out][
                sex[v][preds[v]["qualifiers"][out]] == r
            ]
            df.loc[r, out] = skl.metrics.roc_auc_score(
                y_true=qualified_labels, y_score=qualified_preds
            )
    print(df)

logger.info("---fin")
