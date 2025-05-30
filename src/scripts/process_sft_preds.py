#!/usr/bin/env python3

"""
load results from extract_outcomes and fine_tuned_predictions and
generate summary stats
"""

import argparse
import functools
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import sklearn as skl

from src.framework.logger import get_logger
from src.framework.util import log_classification_metrics, set_pd_options
from src.framework.vocabulary import Vocabulary

set_pd_options()

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../clif-data-ucmc")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_sft_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/mdl-llama1b-57928921-run1-58115722-clsfr-same_admission_death",
)
parser.add_argument(
    "--model_outlier_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama1b-57928921-run1",
)
parser.add_argument(
    "--outcome",
    choices=[
        "same_admission_death",
        "long_length_of_stay",
        "icu_admission",
        "imv_event",
    ],
    default="same_admission_death",
)
parser.add_argument("--only_new", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_sft_loc, model_outlier_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_sft_loc, args.model_outlier_loc),
)
data_version = args.data_version
outcome = args.outcome

versions = ("orig", "new") if not args.only_new else ("new",)
data_dir = dict()
outliers = dict()
label = dict()
sft_pred = dict()
qualifiers = dict()
race = dict()
ethnicity = dict()
sex = dict()

vocab = Vocabulary().load(
    data_dir_orig.joinpath(f"{args.data_version}-tokenized", "train", "vocab.gzip")
)

for v in versions:
    logger.info(f"{v=}")
    data_dir[v] = (data_dir_orig if v == "orig" else data_dir_new).joinpath(
        f"{data_version}-tokenized", "test"
    )
    pred_ = np.load(
        data_dir[v].joinpath(
            "sft-{o}-preds-{m}.npy".format(o=outcome, m=model_sft_loc.stem)
        ),
    )
    if pred_.shape[-1] == 2:
        qualifiers[v] = pred_[:, 1].astype(bool)
        sft_pred[v] = (pred_[:, 0])[qualifiers[v]]
    else:
        sft_pred[v] = pred_
        qualifiers[v] = np.ones_like(pred_).astype(bool)
    outliers[v] = (
        np.load(
            data_dir[v].joinpath(
                "features-outliers-{m}.npy".format(m=model_outlier_loc.stem)
            )
        )  # "Returns -1 for outliers and 1 for inliers"
        == -1
    )[qualifiers[v]]
    label[v] = (
        pl.scan_parquet(data_dir[v].joinpath("tokens_timelines_outcomes.parquet"))
        .select(outcome)
        .collect()
        .to_numpy()
        .ravel()
    )[qualifiers[v]]
    logger.info("For all...")
    log_classification_metrics(y_true=label[v], y_score=sft_pred[v], logger=logger)
    logger.info("For inliers...")
    log_classification_metrics(
        y_true=label[v][~outliers[v]], y_score=sft_pred[v][~outliers[v]], logger=logger
    )
    logger.info("For outliers...")
    log_classification_metrics(
        y_true=label[v][outliers[v]], y_score=sft_pred[v][outliers[v]], logger=logger
    )
    race[v] = (
        pl.scan_parquet(data_dir[v].joinpath("tokens_timelines_outcomes.parquet"))
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
        pl.scan_parquet(data_dir[v].joinpath("tokens_timelines_outcomes.parquet"))
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
        pl.scan_parquet(data_dir[v].joinpath("tokens_timelines_outcomes.parquet"))
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

races = sorted(functools.reduce(set.union, map(set, race.values())))
ethnicities = sorted(functools.reduce(set.union, map(set, ethnicity.values())))
sexes = sorted(functools.reduce(set.union, map(set, sex.values())))

for v in versions:
    print(v.upper().ljust(79, "-"))
    df = pd.DataFrame(index=races, columns=["auc"])
    for r in races:
        df.loc[r] = skl.metrics.roc_auc_score(
            y_true=label[v][(race[v] == r)[qualifiers[v]]],
            y_score=sft_pred[v][(race[v] == r)[qualifiers[v]]],
        )
    print(df)
    df = pd.DataFrame(index=ethnicities, columns=["auc"])
    for r in ethnicities:
        df.loc[r] = skl.metrics.roc_auc_score(
            y_true=label[v][(ethnicity[v] == r)[qualifiers[v]]],
            y_score=sft_pred[v][(ethnicity[v] == r)[qualifiers[v]]],
        )
    print(df)
    df = pd.DataFrame(index=sexes, columns=["auc"])
    for r in sexes:
        df.loc[r] = skl.metrics.roc_auc_score(
            y_true=label[v][(sex[v] == r)[qualifiers[v]]],
            y_score=sft_pred[v][(sex[v] == r)[qualifiers[v]]],
        )
    print(df)

logger.info("---fin")
