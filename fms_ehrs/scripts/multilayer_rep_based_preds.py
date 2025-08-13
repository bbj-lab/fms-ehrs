#!/usr/bin/env python3

"""
make some simple predictions outcomes ~ features
provide some performance breakdowns
"""

import argparse
import collections
import pathlib
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import sklearn as skl
import sklearn.metrics as skl_mets

from fms_ehrs.framework.logger import get_logger, log_classification_metrics
from fms_ehrs.framework.plotting import colors
from fms_ehrs.framework.util import set_pd_options

set_pd_options()

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../data-ucmc")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../data-ucmc")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/llama1b-57928921-run1"
)
parser.add_argument(
    "--outcomes",
    type=str,
    nargs="*",
    default=[
        "same_admission_death",
        "long_length_of_stay",
        "imv_event",
        "icu_admission",
    ],
)
parser.add_argument(
    "--classifier",
    choices=["light_gbm", "logistic_regression"],
    default="logistic_regression",
)
parser.add_argument("--save_preds", action="store_true")
parser.add_argument("--fast", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, out_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.out_dir, args.model_loc),
)

splits = ("train", "val", "test")
versions = ("orig", "new")

data_dirs = collections.defaultdict(dict)
features = collections.defaultdict(dict)
qualifiers = collections.defaultdict(lambda: collections.defaultdict(dict))
labels = collections.defaultdict(lambda: collections.defaultdict(dict))

for v in versions:
    for s in splits:
        data_dirs[v][s] = (data_dir_orig if v == "orig" else data_dir_new).joinpath(
            f"{args.data_version}-tokenized", s
        )
        features[v][s] = np.load(
            data_dirs[v][s].joinpath(
                "features-all-layers-{m}.npy".format(m=model_loc.stem)
            )
        )
        for outcome in args.outcomes:
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

h = features[versions[0]][splits[0]].shape[-1]
preds = collections.defaultdict(lambda: collections.defaultdict(dict))
df = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        (range(h), args.outcomes, versions), names=["level", "outcome", "version"]
    ),
    columns=["AUC"],
)

for level in range(h):
    for outcome in args.outcomes:
        logger.info(outcome.replace("_", " ").upper().ljust(79, "-"))
        Xtrain = (features[versions[0]]["train"][..., level])[
            qualifiers[outcome][versions[0]]["train"]
        ]
        ytrain = (labels[outcome][versions[0]]["train"])[
            qualifiers[outcome][versions[0]]["train"]
        ]
        Xval = (features[versions[0]]["val"][..., level])[
            qualifiers[outcome][versions[0]]["val"]
        ]
        yval = (labels[outcome][versions[0]]["val"])[
            qualifiers[outcome][versions[0]]["val"]
        ]
        match args.classifier:
            case "light_gbm":
                estimator = lgb.LGBMClassifier(metric="auc")
                estimator.fit(X=Xtrain, y=ytrain, eval_set=(Xval, yval))
            case "logistic_regression":
                estimator = skl.pipeline.make_pipeline(
                    skl.preprocessing.StandardScaler(),
                    skl.linear_model.LogisticRegression(
                        max_iter=10 if args.fast else 10_000,
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
        estimator.fit(X=Xtrain, y=ytrain)
        for v in versions:
            q_test = qualifiers[outcome][v]["test"]
            preds[level][outcome][v] = estimator.predict_proba(
                (features[v]["test"][..., level])[q_test]
            )[:, 1]
            y_true = (labels[outcome][v]["test"])[q_test]
            y_score = preds[level][outcome][v]
            logger.info("overall performance".upper().ljust(49, "-"))
            logger.info(
                "{n} qualifying ({p:.2f}%)".format(
                    n=q_test.sum(), p=100 * q_test.mean()
                )
            )
            log_classification_metrics(y_true=y_true, y_score=y_score, logger=logger)
            df.loc[(level, outcome, v)] = skl_mets.roc_auc_score(
                y_true=y_true, y_score=y_score
            )

for v in versions:
    fig = px.scatter(
        df.reset_index().loc[lambda df: df.version == v],
        x="level",
        y="AUC",
        color="outcome",
        width=650,
        title="AUC vs. layer level",
        color_discrete_sequence=colors[1:],
    )
    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        font_family="CMU Serif, Times New Roman, serif",
    )
    fig.update_traces(mode="lines+markers")
    fig.write_image(
        out_dir.joinpath(
            "multilayer-{c}-aucs-ucmc-{m}-{v}{f}.pdf".format(
                c=args.classifier, m=model_loc.stem, v=v, f="-fast" if args.fast else ""
            )
        )
    )

if args.save_preds:
    for v in versions:
        with open(
            data_dirs[v]["test"].joinpath(
                args.classifier
                + "-all_layers-preds-"
                + model_loc.stem
                + ("-fast" if args.fast else "")
                + ".pkl"
            ),
            "wb",
        ) as fp:
            pickle.dump(
                {
                    "qualifiers": {
                        outcome: qualifiers[outcome][v]["test"]
                        for outcome in args.outcomes
                    },
                    "predictions": {
                        outcome: np.stack(
                            [preds[lvl][outcome][v] for lvl in range(h)], axis=-1
                        )
                        for outcome in args.outcomes
                    },
                    "labels": {
                        outcome: labels[outcome][v]["test"] for outcome in args.outcomes
                    },
                },
                fp,
            )

logger.info("---fin")
