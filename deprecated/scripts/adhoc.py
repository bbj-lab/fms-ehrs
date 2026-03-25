#!/usr/bin/env python3

"""
for a list of models, collect predictions and compare performance
"""

import argparse
import pathlib

import polars as pl
import statsmodels.formula.api as smf

from fms_ehrs.framework.logger import get_logger, log_classification_metrics

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument(
    "--outcomes",
    type=str,
    nargs="*",
    default=["same_admission_death", "long_length_of_stay"],
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.out_dir)
)

# load and prep data
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits}

tto_train = (
    pl.read_parquet(data_dirs["train"].joinpath("tokens_timelines_outcomes.parquet"))
    .with_columns(s_len=pl.min_horizontal("seq_len", 1024))
    .cast({out: int for out in args.outcomes})
    .to_pandas()
)
tto_test = (
    pl.read_parquet(data_dirs["test"].joinpath("tokens_timelines_outcomes.parquet"))
    .with_columns(s_len=pl.min_horizontal("seq_len", 1024))
    .cast({out: int for out in args.outcomes})
    .to_pandas()
)
for out in args.outcomes:
    print(out)
    lm_len = smf.logit(f"{out} ~ s_len", data=tto_train).fit()
    logger.info(lm_len.summary())
    preds = lm_len.predict(tto_test)
    log_classification_metrics(y_true=tto_test[out], y_score=preds, logger=logger)


""" query raw tables
"""

df = pl.read_csv(
    "/gpfs/data/bbj-lab/code/divergence/physionet.org/files/mimiciv/2.2/hosp/admissions.csv.gz"
)

with pl.Config(tbl_rows=-1):
    df.group_by("race").len().with_columns(
        pct=pl.col("len") / pl.col("len").sum()
    ).sort("len", descending=True)

with pl.Config(tbl_rows=-1):
    df.with_columns(
        prefix=pl.col("race")
        .str.replace(r"[-/]", "-")
        .str.split("-")
        .list.first()
        .str.strip_chars()
        .replace(
            {
                "HISPANIC OR LATINO": "HISPANIC",
                "UNABLE TO OBTAIN": "UNKNOWN",
                "PATIENT DECLINED TO ANSWER": "UNKNOWN",
                "PORTUGUESE": "WHITE",
                "AMERICAN INDIAN": "OTHER",
                "MULTIPLE RACE": "OTHER",
                "SOUTH AMERICAN": "OTHER",
                "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "OTHER",
            }
        )
    ).group_by("prefix").len().with_columns(
        pct=pl.col("len") / pl.col("len").sum()
    ).sort("len", descending=True)


"""
"""

# df = pl.read_parquet(
#     "/gpfs/data/bbj-lab/users/burkh4rt/data-raw/mimic-2.1.0/clif_respiratory_support.parquet"
# )
#
# df = pl.scan_csv("/gpfs/data/bbj-lab/users/burkh4rt/mimiciv-3.1/icu/chartevents.csv.gz")
#
# df.head(1000).collect()


# import time

# def test1():
#     start = time.perf_counter()
#     pl.scan_csv(
#         "/gpfs/data/bbj-lab/users/burkh4rt/mimiciv-3.1/icu/chartevents.csv.gz"
#     ).head(1000).collect()
#     end = time.perf_counter()
#     return f"Elapsed: {end - start:.6f} seconds"
#
#
# def test2():
#     start = time.perf_counter()
#     pl.scan_csv(
#         "/gpfs/data/bbj-lab/users/burkh4rt/mimiciv-3.1/icu/chartevents.csv.gz",
#         n_rows=1000,
#     ).collect()
#     end = time.perf_counter()
#     return f"Elapsed: {end - start:.6f} seconds"
#
#
# print(test1())
# print(test2())
