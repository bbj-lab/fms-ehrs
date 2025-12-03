#!/usr/bin/env python3

"""
grab the MEDS version of MIMIC
"""

import pathlib

import polars as pl

pl.scan_parquet(
    [
        "/gpfs/data/bbj-lab/users/burkh4rt/ihlee-mimic-meds/train/*.parquet",
        "/gpfs/data/bbj-lab/users/burkh4rt/ihlee-mimic-meds/test/*.parquet",
    ]
).sink_parquet(
    "/gpfs/data/bbj-lab/users/burkh4rt/data-raw/mimic-meds-ihlee/meds.parquet"
)

df = pl.scan_parquet(
    "/gpfs/data/bbj-lab/users/burkh4rt/data-raw/mimic-meds-ihlee/meds.parquet"
)
df.join(
    df.select(pl.col("subject_id").unique())
    .collect()
    .sample(fraction=0.01, with_replacement=False, seed=42)
    .lazy(),
    on="subject_id",
    validate="m:1",
    how="inner",
).sink_parquet(
    "/gpfs/data/bbj-lab/users/burkh4rt/development-sample-21/raw-meds/dev/meds.parquet"
)


dev_dir = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/development-sample-21")
df = pl.read_parquet(dev_dir / "raw-meds" / "dev" / "meds.parquet")


with pl.Config(tbl_rows=30):
    print(
        df.select(pl.col("code").str.split("//").list[0])
        .to_series()
        .value_counts()
        .sort("count", descending=True)
    )
