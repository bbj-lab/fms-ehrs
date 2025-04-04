#!/usr/bin/env python3

"""
some scripts to get us to the point where we can do things
"""

import os
import pathlib

import polars as pl

hm = pathlib.Path("/gpfs/data/bbj-lab/users/{}".format(os.getenv("USER")))
mimic_hm = hm.joinpath("physionet.org/files/mimiciv/3.1/")
uchi_hm = pathlib.Path("/scratch", os.getenv("USER"), "CLIF-2.0.0")


""" restrict to Blood-based labs
"""

import pathlib

import pandas as pd

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")
mimic_orig = hm.joinpath("mimiciv-3.1/")
mimic_edit = hm.joinpath("mimiciv-3.1-edit/")

pd.read_csv(mimic_orig.joinpath("hosp/labevents.csv.gz")).merge(
    pd.read_csv(mimic_orig.joinpath("hosp/d_labitems.csv.gz")).loc[
        lambda df: df.fluid == "Blood", "itemid"
    ],
    on="itemid",
    validate="m:1",
).to_csv(mimic_edit.joinpath("hosp/labevents.csv.gz"), compression="gzip")


""" collect summary statistics for different versions of data
"""

import pathlib

import polars as pl

from tokenizer import ClifTokenizer, summarize

versions = ("day_stays_qc", "QC_day_stays")

for v in versions:
    tpath = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data").joinpath(
        f"{v}-tokenized", "train"
    )
    tt = pl.read_parquet(tpath.joinpath("tokens_timelines.parquet"))
    tknzr = ClifTokenizer(vocab_path=tpath.joinpath("vocab.gzip"))
    summarize(tokenizer=tknzr, tokens_timelines=tt)

versions = ("day_stays_qc", "QC_day_stays")

for v in versions:
    print(v)
    tpath = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data").joinpath(
        f"{v}_first_24h-tokenized", "train"
    )
    tt = pl.read_parquet(tpath.joinpath("tokens_timelines.parquet"))
    tknzr = ClifTokenizer(vocab_path=tpath.joinpath("vocab.gzip"))
    summarize(tokenizer=tknzr, tokens_timelines=tt)


data_dir = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data")
for f in data_dir.joinpath("raw", "train").glob("*.parquet"):
    print(f.stem)
    for v in ("raw", "QC"):
        print(v)
        pl.read_parquet(data_dir.joinpath(v, "train", f.name))

with pl.Config(tbl_rows=100):
    for v in ("raw", "QC"):
        pl.read_parquet(
            data_dir.joinpath(v, "train", "clif_patient_assessments.parquet")
        ).select("assessment_category").to_series().value_counts().sort(
            "assessment_category"
        )


for v in ("QC_day_stays_first_24h", "QC_day_stays"):
    for s in ("train", "val", "test"):
        tpath = pathlib.Path("/scratch/burkh4rt/clif-data").joinpath(
            f"{v}-tokenized", s
        )
        print(pl.read_parquet(tpath.joinpath("tokens_timelines.parquet")))


for v in ("QC_day_stays_first_24h", "QC_day_stays"):
    for s in ("train", "val", "test"):
        tpath = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data").joinpath(
            f"{v}-tokenized", s
        )
        print(pl.read_parquet(tpath.joinpath("tokens_timelines.parquet")))

""" get covid-ish patients
"""

covid_hids = (
    pl.read_csv(mimic_hm.joinpath("hosp/patients.csv.gz"))
    .filter(pl.col("anchor_year_group") == "2020 - 2022")
    .cast({"subject_id": str})
    .join(
        pl.read_csv(mimic_hm.joinpath("hosp/admissions.csv.gz")).cast(
            {"subject_id": str}
        ),
        on="subject_id",
        how="inner",
        validate="1:m",
    )
    .select(pl.col("hadm_id").cast(str).alias("hospitalization_id"))
)
