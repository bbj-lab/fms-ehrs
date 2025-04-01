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
import sklearn as skl

from logger import get_logger
from util import log_classification_metrics, set_pd_options

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
parser.add_argument(
    "--classifier",
    choices=["light_gbm", "logistic_regression_cv", "logistic_regression"],
    default="logistic_regression",
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
outcomes = ("same_admission_death", "long_length_of_stay")
data_dirs = collections.defaultdict(dict)
outliers = collections.defaultdict(dict)
icus = collections.defaultdict(dict)
features = collections.defaultdict(dict)
labels = collections.defaultdict(lambda: collections.defaultdict(dict))

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
        features[v][s] = np.load(
            data_dirs[v][s].joinpath("features-{m}.npy".format(m=model_loc.stem))
        )
        for outcome in outcomes:
            labels[outcome][v][s] = (
                pl.scan_parquet(
                    data_dirs[v][s].joinpath("tokens_timelines_outcomes.parquet")
                )
                .select(outcome)
                .collect()
                .to_numpy()
                .ravel()
            )
        icus[v][s] = (
            pl.scan_parquet(
                data_dirs[v][s].joinpath("tokens_timelines_outcomes.parquet")
            )
            .select("icu_stay")
            .collect()
            .to_numpy()
            .ravel()
        )

""" classification outcomes
"""

preds = collections.defaultdict(dict)

for outcome in outcomes:
    logger.info(outcome.replace("_", " ").upper().ljust(79, "-"))

    match args.classifier:
        case "light_gbm":
            estimator = lgb.LGBMClassifier(
                metric="auc",
                force_col_wise=True,
                learning_rate=0.05 if not fast else 0.1,
                n_estimators=1000 if not fast else 100,
            )
            estimator.fit(
                X=features["orig"]["train"],
                y=labels[outcome]["orig"]["train"],
                eval_set=(features["orig"]["val"], labels[outcome]["orig"]["val"]),
            )

        case "logistic_regression_cv":
            estimator = skl.pipeline.make_pipeline(
                skl.preprocessing.StandardScaler(),
                skl.linear_model.LogisticRegressionCV(
                    max_iter=10_000 if not fast else 100,
                    n_jobs=-1,
                    refit=True,
                    random_state=42,
                    solver="newton-cholesky" if not fast else "lbfgs",
                ),
            )
            estimator.fit(
                X=features["orig"]["train"], y=labels[outcome]["orig"]["train"]
            )

        case "logistic_regression":
            estimator = skl.pipeline.make_pipeline(
                skl.preprocessing.StandardScaler(),
                skl.linear_model.LogisticRegression(
                    max_iter=10_000 if not fast else 100,
                    n_jobs=-1,
                    random_state=42,
                    solver="newton-cholesky" if not fast else "lbfgs",
                ),
            )
            estimator.fit(
                X=features["orig"]["train"], y=labels[outcome]["orig"]["train"]
            )

        case _:
            raise NotImplementedError(
                f"Classifier {args.classifier} is not yet supported."
            )

    for v in versions:
        logger.info(v.upper())
        preds[outcome][v] = estimator.predict_proba(features[v]["test"])[:, 1]
        y_true = labels[outcome][v]["test"]
        y_score = preds[outcome][v]
        logger.info("overall performance".upper().ljust(49, "-"))
        log_classification_metrics(y_true=y_true, y_score=y_score, logger=logger)
        logger.info("on outliers".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=y_true[outliers[v]["test"]],
            y_score=y_score[outliers[v]["test"]],
            logger=logger,
        )
        logger.info("on inliers".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=y_true[~outliers[v]["test"]],
            y_score=y_score[~outliers[v]["test"]],
            logger=logger,
        )
        logger.info("for icu".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=y_true[icus[v]["test"]],
            y_score=y_score[icus[v]["test"]],
            logger=logger,
        )
        logger.info("for non-icu".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=y_true[~icus[v]["test"]],
            y_score=y_score[~icus[v]["test"]],
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
