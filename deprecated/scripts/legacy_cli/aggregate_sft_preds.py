#!/usr/bin/env python3

"""
for a list of models, collect fine-tuned predictions and compare performance
"""

import argparse
import collections
import pathlib

import numpy as np
import polars as pl

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
    "--outcome",
    choices=[
        "same_admission_death",
        "long_length_of_stay",
        "icu_admission",
        "imv_event",
    ],
    required=True,
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

y_true = (
    pl.scan_parquet(data_dir_test.joinpath("tokens_timelines_outcomes.parquet"))
    .select(args.outcome)
    .collect()
    .to_numpy()
    .ravel()
    .astype(int)
)

named_results = collections.OrderedDict()
for m in models:
    scr_qual = np.load(
        data_dir_test.joinpath("sft-{o}-preds-{m}.npy".format(o=args.outcome, m=m.stem))
    )
    named_results[m.stem.split("-")[2]] = {
        "y_true": y_true[scr_qual[:, 1].astype(bool).ravel()],
        "y_score": scr_qual[scr_qual[:, 1].astype(bool).ravel(), 0].ravel(),
    }


plot_calibration_curve(
    named_results,
    savepath=out_dir.joinpath(f"sft-cal-{args.outcome}-{data_dir.stem}.pdf"),
)
plot_roc_curve(
    named_results,
    savepath=out_dir.joinpath(f"sft-roc-{args.outcome}-{data_dir.stem}.pdf"),
)
plot_precision_recall_curve(
    named_results,
    savepath=out_dir.joinpath(f"sft-pr-{args.outcome}-{data_dir.stem}.pdf"),
)

logger.info("---fin")
