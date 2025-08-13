#!/usr/bin/env python3

"""
for a list of models, collect predictions and compare performance
"""

import argparse
import collections
import pathlib
import pickle

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import (
    plot_calibration_curve,
    plot_precision_recall_curve,
    plot_roc_curve,
)

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path)
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument("--out_dir", type=pathlib.Path)
parser.add_argument(
    "--classifier",
    choices=["light_gbm", "logistic_regression_cv", "logistic_regression"],
    default="logistic_regression",
)
parser.add_argument("--models", nargs="*")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir, *models = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.out_dir, *args.models),
)
data_dir_test = data_dir.joinpath(f"{args.data_version}-tokenized", "test")
outcomes = ("same_admission_death", "long_length_of_stay", "icu_admission", "imv_event")

results = collections.OrderedDict()
for m in models:
    with open(
        data_dir_test.joinpath(args.classifier + "-preds-" + m.stem + ".pkl"), "rb"
    ) as fp:
        results[m.stem] = pickle.load(fp)

for outcome in outcomes:
    named_results = collections.OrderedDict()
    for m, v in results.items():
        named_results[m.split("-")[1]] = {
            "y_true": v["labels"][outcome][v["qualifiers"][outcome]],
            "y_score": v["predictions"][outcome],
        }
    plot_calibration_curve(
        named_results, savepath=out_dir.joinpath(f"cal-{outcome}-{data_dir.stem}.pdf")
    )
    plot_roc_curve(
        named_results, savepath=out_dir.joinpath(f"roc-{outcome}-{data_dir.stem}.pdf")
    )
    plot_precision_recall_curve(
        named_results, savepath=out_dir.joinpath(f"pr-{outcome}-{data_dir.stem}.pdf")
    )

logger.info("---fin")
