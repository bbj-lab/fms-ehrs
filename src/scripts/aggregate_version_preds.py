#!/usr/bin/env python3

"""
for a list of data versions, collect predictions and compare performance
"""


import argparse
import collections
import pathlib
import pickle
import re

from src.framework.logger import get_logger
from src.framework.util import (
    plot_calibration_curve,
    plot_precision_recall_curve,
    plot_roc_curve,
)

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../clif-data")
parser.add_argument(
    "--data_versions",
    type=str,
    nargs="*",
    default=[
        "icu24h_first_24h",
        "icu24h_top5-921_first_24h",
        "icu24h_bot5-921_first_24h",
        "icu24h_rnd5-921_first_24h",
    ],
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument(
    "--classifier",
    choices=["light_gbm", "logistic_regression_cv", "logistic_regression"],
    default="logistic_regression",
)
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama1b-57928921-run1",
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.out_dir, args.model_loc),
)

outcomes = ("same_admission_death", "long_length_of_stay", "imv_event")

results = collections.OrderedDict()
for v in args.data_versions:
    with open(
        data_dir.joinpath(
            f"{v}-tokenized",
            "test",
            args.classifier + "-preds-" + model_loc.stem + ".pkl",
        ),
        "rb",
    ) as fp:
        results[v] = pickle.load(fp)

for outcome in outcomes:
    named_results = collections.OrderedDict()
    for m, v in results.items():
        des = re.split("[-_]", m)[1] if m != "icu24h_first_24h" else "orig"
        named_results[des] = {
            "y_true": v["labels"][outcome][v["qualifiers"][outcome]],
            "y_score": v["predictions"][outcome],
        }
    plot_calibration_curve(
        named_results,
        savepath=out_dir.joinpath(f"cal-{outcome}-{data_dir.stem}.pdf"),
    )
    plot_roc_curve(
        named_results,
        savepath=out_dir.joinpath(f"roc-{outcome}-{data_dir.stem}.pdf"),
    )
    plot_precision_recall_curve(
        named_results,
        savepath=out_dir.joinpath(f"pr-{outcome}-{data_dir.stem}.pdf"),
    )

logger.info("---fin")
