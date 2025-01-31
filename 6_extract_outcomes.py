#!/usr/bin/env python3

"""
determine some outcomes of interest for each hospitalization
"""

import pathlib

import polars as pl

from tokenizer import ClifTokenizer
from vocabulary import Vocabulary

data_version = "all"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

# load and prep data
splits = ("train", "val", "test")
data_dirs = dict()
ref_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath("clif-data", "raw", s)
    ref_dirs[s] = hm.joinpath("clif-data", "first-24h-tokenized", s)

vocab = Vocabulary().load(
    hm.joinpath("clif-data", "day-stays-tokenized", "train", "vocab.gzip")
)

for s in splits:
    outcomes = (
        ClifTokenizer(
            data_dir=data_dirs[s],
            vocab_path=hm.joinpath(
                "clif-data", "day-stays-tokenized", "train", "vocab.gzip"
            ),
        )
        .get_tokens_timelines()
        .lazy()
        .with_columns(
            length_of_stay=(
                pl.col("times").list.max() - pl.col("times").list.min()
            ).dt.total_hours(),
            # length_of_stay_qc=(
            #     pl.col("times").list.get(-1) - pl.col("times").list.get(0)
            # ).dt.total_hours(),
            same_admission_death=pl.col("tokens").list.contains(vocab("expired")),
        )
        .select("hospitalization_id", "length_of_stay", "same_admission_death")
    )
    (
        pl.scan_parquet(ref_dirs[s].joinpath("tokens_timelines.parquet"))
        .select("hospitalization_id")
        .join(outcomes, how="left", on="hospitalization_id")
        .sink_parquet(ref_dirs[s].joinpath("outcomes.parquet"))
    )
