#!/usr/bin/env python3

"""
obtain ICU types associated to the first ICU admission for each 
hospitalization_id in MIMIC
"""

import pathlib

import polars as pl

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")
mimic_hm = hm.joinpath("physionet.org/files/mimiciv/3.1/")

(
    pl.scan_csv(mimic_hm.joinpath("hosp", "transfers.csv.gz"))
    .filter(
        ~pl.col("hadm_id").is_null()
        & (pl.col("eventtype") == "admit")
        & (pl.col("careunit").str.contains("ICU"))
    )
    .sort(pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"))
    .group_by("hadm_id", maintain_order=True)
    .agg(pl.col("careunit").first())
    .collect()
    .write_csv(hm.joinpath("mimic_icu_types.csv.gz"))
)
