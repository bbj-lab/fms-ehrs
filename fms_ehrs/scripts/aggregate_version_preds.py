#!/usr/bin/env python3

"""
for a list of data versions, collect predictions and compare performance
"""

import argparse
import collections
import pathlib
import pickle

import numpy as np
import pandas as pd
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import (
    plot_calibration_curve,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from fms_ehrs.framework.stats import bootstrap_ci, bootstrap_pval

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
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
parser.add_argument(
    "--handles", type=str, nargs="*", default=["orig", "top5", "bot5", "rnd5"]
)
parser.add_argument("--baseline_handle", type=str, default="orig")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument(
    "--classifier",
    choices=["light_gbm", "logistic_regression_cv", "logistic_regression"],
    default="logistic_regression",
)
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--suffix", type=str, default="")
parser.add_argument(
    "--outcomes",
    type=str,
    nargs="*",
    default=["same_admission_death", "long_length_of_stay"],
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.out_dir, args.model_loc),
)

lookup = dict(zip(args.data_versions, args.handles))

results = collections.OrderedDict()
tto = collections.OrderedDict()
for v in args.data_versions:
    logger.info(f"{v=}")
    with open(
        data_dir.joinpath(
            f"{v}-tokenized",
            "test",
            args.classifier + "-preds-" + model_loc.stem + ".pkl",
        ),
        "rb",
    ) as fp:
        results[v] = pickle.load(fp)
    tto[v] = pl.scan_parquet(
        data_dir.joinpath(f"{v}-tokenized", "test", "tokens_timelines_outcomes.parquet")
    )
    # logger.info(tto[v].select(pl.col("times").list.len()).describe())

suffix = ("-" + args.suffix) if args.suffix != "" else ""
for outcome in args.outcomes:
    logger.info(outcome.upper().ljust(79, "-"))
    named_results = collections.OrderedDict()
    for k, v in results.items():
        named_results[lookup[k]] = {
            "y_true": v["labels"][outcome][v["qualifiers"][outcome]],
            "y_score": v["predictions"][outcome],
        }
    plot_calibration_curve(
        named_results,
        savepath=out_dir.joinpath(
            f"cal-{outcome}-{data_dir.stem}-{model_loc.stem}{suffix}.pdf"
        ),
    )
    plot_roc_curve(
        named_results,
        savepath=out_dir.joinpath(
            f"roc-{outcome}-{data_dir.stem}-{model_loc.stem}{suffix}.pdf"
        ),
    )
    plot_precision_recall_curve(
        named_results,
        savepath=out_dir.joinpath(
            f"pr-{outcome}-{data_dir.stem}-{model_loc.stem}{suffix}.pdf"
        ),
    )

    results_tbl = pd.DataFrame(
        columns=["avg-len", "CI-roc_auc", "CI-pr_auc", "CI-brier"],
        index=pd.Index(named_results.keys(), name="versions"),
    )

    results_alt = pd.DataFrame(
        columns=["roc_auc", "pr_auc", "brier"],
        index=pd.Index(named_results.keys(), name="versions"),
    )

    for name, res in named_results.items():
        cis = bootstrap_ci(
            res["y_true"],
            res["y_score"],
            n_samples=10_0,
            objs=("roc_auc", "pr_auc", "brier"),
        )
        for k, v in cis.items():
            results_tbl.loc[name, f"CI-{k}"] = tuple(v.round(3).tolist())
            results_alt.loc[name, k] = "${m} \pm {d}$".format(
                m=(m := v.mean().round(3)), d=np.max(np.abs(v - m)).round(3)
            )

    results_tbl["avg-len"] = [
        tto[v].select(pl.col("times").list.len().mean()).collect().item()
        for v in args.data_versions
    ]

    logger.info(results_tbl)
    logger.info(results_alt.to_latex())

    p_tbl = pd.DataFrame(
        columns=["roc_auc-p", "pr_auc-p", "brier-p"],
        index=pd.Index(
            [n for n in named_results.keys() if n != args.baseline_handle],
            name="versions",
        ),
    )
    for name, res in named_results.items():
        if name != args.baseline_handle:
            pvals = bootstrap_pval(
                res["y_true"],
                res["y_score"],
                named_results[args.baseline_handle]["y_score"],
            )
            p_tbl.loc[name] = {f"{k}-p": v for k, v in pvals.items()}

    logger.info(p_tbl)


logger.info("---fin")
