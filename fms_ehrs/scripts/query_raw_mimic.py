#!/usr/bin/env python3

"""
look up relevant info for provided hospitalization id's in MIMIC
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
    "--mimic_dir",
    type=pathlib.Path,
    default="../../mimiciv-3.1",
    help="""The directory containing the hosp & icu directories downloaded from
    physionet.""",
)
parser.add_argument(
    "--hadm_ids",
    type=str,
    nargs="*",
    default=["24640534", "26886976"],
)


args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

mimic_dir = pathlib.Path(args.mimic_dir).expanduser().resolve()
mimic_tables = {
    "{d}-{n}".format(d=csv.parent.stem, n=csv.stem.replace(".csv", "")): pl.scan_csv(
        csv, infer_schema=False
    )
    for csv in mimic_dir.rglob("*.csv.gz")
}
mimic_tables["hosp-diagnoses_icd"] = mimic_tables["hosp-diagnoses_icd"].join(
    mimic_tables["hosp-d_icd_diagnoses"],
    on=["icd_code", "icd_version"],
    how="full",
    validate="m:1",
)

for hadm_id in args.hadm_ids:
    print(hadm_id.center(79, "="))
    print(
        mimic_tables["hosp-admissions"]
        .filter(pl.col("hadm_id") == hadm_id)
        .collect()
        .to_numpy()
    )
    current_dx = (
        mimic_tables["hosp-diagnoses_icd"]
        .filter(pl.col("hadm_id") == hadm_id)
        .select("icd_code", "long_title")
    )
    print(current_dx.collect().to_numpy())
    adm = (
        mimic_tables["hosp-admissions"]
        .filter(pl.col("hadm_id") == hadm_id)
        .with_columns(
            pl.col("admittime").str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S"),
            pl.col("dischtime").str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S"),
        )
        .with_columns((pl.col("dischtime") - pl.col("admittime")).alias("duration"))
    )
    print(adm.select("hospital_expire_flag", "duration").collect())
    print(
        adm.join(
            mimic_tables["hosp-patients"].select(
                "subject_id", "anchor_age", "anchor_year"
            ),
            on="subject_id",
            validate="1:1",
        )
        .select(
            (
                pl.col("anchor_age").cast(int)
                + pl.col("admittime").dt.year()
                - pl.col("anchor_year").cast(int)
            ).alias("est_age")
        )
        .collect()
    )
    whenami = adm.select("admittime").collect().item()
    whoami = (
        mimic_tables["hosp-admissions"]
        .filter(pl.col("hadm_id") == hadm_id)
        .select("subject_id")
        .collect()
        .item()
    )
    prev = (
        mimic_tables["hosp-admissions"]
        .filter(pl.col("subject_id") == whoami)
        .filter(
            pl.col("admittime").str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S")
            < whenami
        )
    )
    print(prev.collect().to_numpy())
    prev_dx = (
        mimic_tables["hosp-diagnoses_icd"]
        .join(prev, on="hadm_id", how="inner", validate="m:1")
        .select("icd_code", "long_title")
    )
    print("previous".center(79, "-"))
    print(prev_dx.collect().to_numpy())
    print("new".center(79, "-"))
    print(
        current_dx.join(prev_dx, on=["icd_code", "long_title"], how="anti")
        .collect()
        .to_numpy()
    )
