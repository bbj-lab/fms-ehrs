#!/usr/bin/env python3

"""
add tabular data to the train-val-test split after the initial split has been made
"""

import argparse
import pathlib

import polars as pl

from fms_ehrs.framework.logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--new_data_loc",
    type=pathlib.Path,
    default="../../clif-data/scratch/en_ner_bc5cdr_md_umls.csv",
)
parser.add_argument("--new_data_name", type=str, default="machine_measurements")
parser.add_argument("--data_dir_out", type=pathlib.Path, default="../../clif-data/")
parser.add_argument("--data_version", type=str, default="raw")
parser.add_argument("--patient_id_col", type=str, default="subject_id")
parser.add_argument("--time_col", type=str, default="ecg_time")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")


new_data_loc, data_dir_out = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.new_data_loc, args.data_dir_out),
)

new_data = (
    pl.scan_csv(new_data_loc)
    .with_columns(
        pl.col(args.patient_id_col).cast(str).alias("patient_id"),
        pl.col(args.time_col).str.to_datetime().alias("event_dttm").cast(pl.Datetime),
    )
    .with_row_index("new_idx")
)

# get sub-directories
splits = ("train", "val", "test")
for s in splits:
    dir_out = data_dir_out.joinpath(args.data_version, s)
    collated = (
        pl.scan_parquet(dir_out.joinpath("clif_hospitalization.parquet"))
        .select(
            "patient_id",
            "hospitalization_id",
            pl.col("admission_dttm").dt.replace_time_zone(None),
            pl.col("discharge_dttm").dt.replace_time_zone(None),
        )
        .join(new_data, on="patient_id")
        .filter(pl.col("event_dttm").is_between("admission_dttm", "discharge_dttm"))
        .drop("patient_id", "admission_dttm", "discharge_dttm")
        .collect()
    )
    logger.info(f"Adding {collated.shape[0]} events to {s} split...")
    if (i := collated.shape[0] - collated.select("new_idx").n_unique()) > 0:
        logger.info(f"Detected {i} overlapping hospitalization(s).")
    collated.drop("new_idx").write_parquet(
        dir_out.joinpath(args.new_data_name + ".parquet")
    )

logger.info("---fin")
