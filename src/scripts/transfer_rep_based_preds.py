#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
break down performance by ICU admission type
"""

import argparse
import collections
import pathlib
import pickle

import lightgbm as lgb
import numpy as np
import polars as pl
import sklearn as skl

from src.framework.logger import get_logger
from src.framework.util import log_classification_metrics, set_pd_options

set_pd_options()

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path)
parser.add_argument("--data_dir_new", type=pathlib.Path)
parser.add_argument("--data_version", type=str)
parser.add_argument("--model_loc", type=pathlib.Path)
parser.add_argument(
    "--classifier",
    choices=["light_gbm", "logistic_regression_cv", "logistic_regression"],
    default="logistic_regression",
)
parser.add_argument("--save_preds", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc),
)

splits = ("train", "val", "test")
versions = ("orig", "new")
outcomes = ("same_admission_death", "long_length_of_stay", "icu_admission", "imv_event")

data_dirs = collections.defaultdict(dict)
outliers = collections.defaultdict(dict)
features = collections.defaultdict(dict)
qualifiers = collections.defaultdict(lambda: collections.defaultdict(dict))
labels = collections.defaultdict(lambda: collections.defaultdict(dict))

for v in versions:
    for s in splits:
        data_dirs[v][s] = (data_dir_orig if v == "orig" else data_dir_new).joinpath(
            f"{args.data_version}-tokenized", s
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
            qualifiers[outcome][v][s] = (
                (
                    ~pl.scan_parquet(
                        data_dirs[v][s].joinpath("tokens_timelines_outcomes.parquet")
                    )
                    .select(outcome + "_24h")
                    .collect()
                    .to_numpy()
                    .ravel()
                )  # *not* people who have had this outcome in the first 24h
                if outcome in ("icu_admission", "imv_event")
                else True * np.ones_like(labels[outcome][v][s])
            )


""" classification outcomes
"""

preds = collections.defaultdict(dict)

for outcome in outcomes:

    logger.info(outcome.replace("_", " ").upper().ljust(79, "-"))

    Xtrain = (features["orig"]["train"])[qualifiers[outcome]["orig"]["train"]]
    ytrain = (labels[outcome]["orig"]["train"])[qualifiers[outcome]["orig"]["train"]]
    Xval = (features["orig"]["val"])[qualifiers[outcome]["orig"]["val"]]
    yval = (labels[outcome]["orig"]["val"])[qualifiers[outcome]["orig"]["val"]]

    match args.classifier:
        case "light_gbm":
            estimator = lgb.LGBMClassifier(metric="auc")
            estimator.fit(
                X=Xtrain,
                y=ytrain,
                eval_set=(Xval, yval),
            )

        case "logistic_regression_cv":
            estimator = skl.pipeline.make_pipeline(
                skl.preprocessing.StandardScaler(),
                skl.linear_model.LogisticRegressionCV(
                    max_iter=10_000,
                    n_jobs=-1,
                    refit=True,
                    random_state=42,
                    solver="newton-cholesky",
                ),
            )
            estimator.fit(X=Xtrain, y=ytrain)

        case "logistic_regression":
            estimator = skl.pipeline.make_pipeline(
                skl.preprocessing.StandardScaler(),
                skl.linear_model.LogisticRegression(
                    max_iter=10_000,
                    n_jobs=-1,
                    random_state=42,
                    solver="newton-cholesky",
                ),
            )
            estimator.fit(X=Xtrain, y=ytrain)

        case _:
            raise NotImplementedError(
                f"Classifier {args.classifier} is not yet supported."
            )
    for v in versions:

        logger.info(v.upper())

        q_test = qualifiers[outcome][v]["test"]
        preds[outcome][v] = estimator.predict_proba((features[v]["test"])[q_test])[:, 1]
        y_true = (labels[outcome][v]["test"])[q_test]
        y_score = preds[outcome][v]

        logger.info("overall performance".upper().ljust(49, "-"))
        logger.info(
            "{n} qualifying ({p:.2f}%)".format(n=q_test.sum(), p=100 * q_test.mean())
        )
        log_classification_metrics(y_true=y_true, y_score=y_score, logger=logger)

        out_test = (outliers[v]["test"])[q_test]
        logger.info("on outliers".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=y_true[out_test],
            y_score=y_score[out_test],
            logger=logger,
        )
        logger.info("on inliers".upper().ljust(49, "-"))
        log_classification_metrics(
            y_true=y_true[~out_test],
            y_score=y_score[~out_test],
            logger=logger,
        )

if args.save_preds:
    for v in versions:
        with open(
            data_dirs[v]["test"].joinpath(
                args.classifier + "-preds-" + model_loc.stem + ".pkl"
            ),
            "wb",
        ) as fp:
            pickle.dump(
                {
                    "qualifiers": {
                        outcome: qualifiers[outcome][v]["test"] for outcome in outcomes
                    },
                    "predictions": {outcome: preds[outcome][v] for outcome in outcomes},
                    "labels": {
                        outcome: labels[outcome][v]["test"] for outcome in outcomes
                    },
                },
                fp,
            )

logger.info("---fin")
