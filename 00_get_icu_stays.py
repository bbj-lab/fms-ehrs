#!/usr/bin/env python3

"""
collection hospitalization id's with an ICU stay
"""

import os
import pathlib

import polars as pl

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")
mimic_hm = hm.joinpath("physionet.org/files/mimiciv/3.1/")
uchi_hm = pathlib.Path("/scratch", os.getenv("USER"), "CLIF-2.0.0")

(
    pl.scan_csv(mimic_hm.joinpath("hosp", "transfers.csv.gz"))
    .filter(
        ~pl.col("hadm_id").is_null()
        & (pl.col("eventtype") == "admit")
        & (pl.col("careunit").str.contains("ICU"))
    )
    .select("hadm_id")
    .unique()
    .sink_parquet(hm.joinpath("mimiciv-3.1-icu-hids.parquet"))
)

(
    pl.scan_parquet(uchi_hm.joinpath("clif_adt.parquet"))
    .filter(pl.col("location_category") == "icu")
    .select("hospitalization_id")
    .unique()
    .sink_parquet(hm.joinpath("ucmc-icu-hids.parquet"))
)

