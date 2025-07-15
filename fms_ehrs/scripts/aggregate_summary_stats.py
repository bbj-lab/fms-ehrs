#!/usr/bin/env python3

"""
generate summary statistics for cohorts
"""

import argparse
import pathlib

import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument("--raw_version", type=str, default="raw")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

splits = ("train", "val", "test")
data_dir = pathlib.Path(args.data_dir).expanduser().resolve()

data_dirs = {s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits}
raw_dirs = {s: data_dir.joinpath(args.raw_version, s) for s in splits}
vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

aggregations = [
    pl.col("hospitalization_id").count().alias("count"),
    pl.col("seq_len").mean(),
    pl.col("age_at_admission").mean().alias("avg_age"),
    pl.col("age_at_admission").median().alias("med_age"),
    (pl.col("sex_category") == "Female").mean().alias("pct_female"),
    (pl.col("race_category") == "Black or African American")
    .mean()
    .alias("pct_African_Amer"),
    (pl.col("race_category") == "Asian").mean().alias("pct_Asian"),
    (pl.col("race_category") == "American Indian or Alaska Native")
    .mean()
    .alias("pct_Native_American"),  # Alaska is in America
    (pl.col("race_category") == "Native Hawaiian or Other Pacific Islander")
    .mean()
    .alias("pct_Pacific_Islander"),  # Hawaii is a Pacific island
    pl.col("race_category").is_in(["Unknown", "Other"]).mean().alias("Unknown/Other"),
    (pl.col("ethnicity_category") == "Hispanic").mean().alias("pct_hispanic"),
    (pl.col("length_of_stay") / 24).mean().alias("avg. length of stay (days)"),
    pl.col("same_admission_death").mean(),
    pl.col("long_length_of_stay").mean(),
    pl.col("icu_admission_24h").mean(),
    pl.col("icu_admission").mean(),
    pl.col("imv_event_24h").mean(),
    pl.col("imv_event").mean(),
]


def summarize_split(s):
    tto = pl.read_parquet(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
    hid_ref = pl.read_parquet(
        raw_dirs[s].joinpath("clif_hospitalization.parquet")
    ).select("hospitalization_id", "age_at_admission", "patient_id")
    pat_ref = (
        pl.read_parquet(raw_dirs[s].joinpath("clif_patient.parquet"))
        .select(
            "patient_id",
            "race_category",
            "ethnicity_category",
            "sex_category",
        )
        .group_by("patient_id")
        .first()
    )
    tto_aux = tto.join(
        hid_ref, on="hospitalization_id", validate="1:1", maintain_order="left"
    ).join(pat_ref, on="patient_id", validate="m:1", maintain_order="left")
    return tto_aux.select(*aggregations).insert_column(0, pl.lit(s))


summary = pl.concat(summarize_split(s) for s in splits)

with pl.Config(tbl_cols=len(aggregations) + 3):
    logger.info(summary)

logger.info(
    summary.to_pandas().set_index(["literal"]).transpose().to_latex(float_format="%.3f")
)

logger.info("---fin")
