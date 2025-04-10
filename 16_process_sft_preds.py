#!/usr/bin/env python3

"""
load results from 06_extract_outcomes and 10_fine_tuned_predictions and
generate summary stats
"""

import argparse
import pathlib

import numpy as np
import polars as pl

from logger import get_logger
from util import log_classification_metrics

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path)
parser.add_argument("--data_dir_new", type=pathlib.Path)
parser.add_argument("--data_version", type=str)
parser.add_argument("--model_sft_loc", type=pathlib.Path)
parser.add_argument("--model_outlier_loc", type=pathlib.Path)
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
parser.add_argument("--only_new", type=bool, default=False)
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
