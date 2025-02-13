#!/usr/bin/env python3

"""
determine some outcomes of interest for each hospitalization
"""

import pathlib

import polars as pl

from vocabulary import Vocabulary

ref_version = "day_stays_qc"
data_version = f"{ref_version}_first_24h"

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

# load and prep data
splits = ("train", "val", "test")
data_dirs = dict()
ref_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath("clif-data", f"{data_version}-tokenized", s)
    ref_dirs[s] = hm.joinpath("clif-data", f"{ref_version}-tokenized", s)

vocab = Vocabulary().load(ref_dirs["train"].joinpath("vocab.gzip"))

for s in splits:
    outcomes = (
        pl.scan_parquet(ref_dirs[s].joinpath("tokens_timelines.parquet"))
        .with_columns(
            length_of_stay=(
                pl.col("times").list.get(-1) - pl.col("times").list.get(0)
            ).dt.total_hours(),
            same_admission_death=pl.col("tokens").list.contains(vocab("expired")),
        )
        .select("hospitalization_id", "length_of_stay", "same_admission_death")
    )
    (
        pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines.parquet"))
        .select("hospitalization_id")
        .join(
            outcomes,
            how="left",
            on="hospitalization_id",
            validate="1:1",
            maintain_order="left",
        )
        .collect()
        .write_parquet(data_dirs[s].joinpath("outcomes.parquet"))
    )
