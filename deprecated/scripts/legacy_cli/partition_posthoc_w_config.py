#!/usr/bin/env python3

"""
add tabular data to the train-val-test split after the initial split has been made
"""

import argparse
import pathlib

import polars as pl
import ruamel.yaml as yaml

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--new_data_loc",
    type=pathlib.Path,
    default="../../data-raw/mimic-2.1.0/clif_respiratory_support_processed.parquet",
)
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic/")
parser.add_argument("--data_version", type=str, default="W21")
parser.add_argument(
    "--config_loc", type=pathlib.Path, default="../fms_ehrs/config/clif-21.yaml"
)
parser.add_argument("--development_sample", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

config = (
    yaml.YAML(typ="safe").load(pathlib.Path(args.config_loc).expanduser().resolve())
    if args.config_loc is not None
    else dict()
)

config_ref = config.get("reference", {})
ref_tbl_str = config_ref.get("table", "clif_hospitalization")
sbj_id_str = config.get("subject_id", "hospitalization_id")
grp_id_str = config.get("group_id", "patient_id")

new_data_loc, config_loc, data_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.new_data_loc, args.config_loc, args.data_dir),
)

new_data = (
    pl.scan_parquet(new_data_loc)
    if new_data_loc.suffix == ".parquet"
    else pl.scan_csv(new_data_loc)
)

new_cols = new_data.collect_schema().names()
new_data = (
    new_data.with_columns(pl.col("subject_id").cast(pl.String).alias(sbj_id_str))
    if "subject_id" in new_cols and sbj_id_str not in new_cols
    else new_data.with_columns(pl.col(sbj_id_str).cast(pl.String))
)

splits = ("train", "val", "test") if not args.development_sample else ("dev",)
for s in splits:
    sbj_ids = pl.scan_parquet(
        data_dir.joinpath(args.data_version, s, f"{ref_tbl_str}.parquet")
    ).select(pl.col(sbj_id_str).cast(pl.String))
    set_perms(
        new_data.join(sbj_ids, how="inner", on=sbj_id_str, validate="m:1").sink_parquet
    )(data_dir.joinpath(args.data_version, s, f"{new_data_loc.stem}.parquet"))
    logger.info(f"Added {new_data_loc.stem} to {s} split.")


logger.info("---fin")
