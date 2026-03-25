#!/usr/bin/env python3

"""
grab the MEDS version of MIMIC
"""

import pathlib

import polars as pl

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")

for version in ["mimic-meds-ihlee", "mimic-meds-ed-ihlee"]:
    pl.scan_parquet(
        [hm / version / "train/*.parquet", hm / version / "test/*.parquet"]
    ).sink_parquet(hm / "data-raw" / version / "meds.parquet")

for version, designator in {
    "mimic-meds-ihlee": "raw-meds",
    "mimic-meds-ed-ihlee": "raw-meds-ed",
}.items():
    df = pl.scan_parquet(hm / "data-raw" / version / "meds.parquet")
    df.join(
        df.select(pl.col("subject_id").unique())
        .collect()
        .sample(fraction=0.01, with_replacement=False, seed=42)
        .lazy(),
        on="subject_id",
        validate="m:1",
        how="inner",
    ).sink_parquet(hm / "development-sample-21" / designator / "dev/meds.parquet")


for designator in ["raw-meds", "raw-meds-ed"]:
    print(designator)
    with pl.Config(tbl_rows=-1):
        print(
            pl.read_parquet(
                hm / "development-sample-21" / designator / "dev" / "meds.parquet"
            )
            .select(pl.col("code").str.split("//").list[0])
            .to_series()
            .value_counts()
            .sort("count", descending=True)
        )
