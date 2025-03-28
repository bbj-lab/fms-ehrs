#!/usr/bin/env python3

"""
determine some outcomes of interest for each hospitalization
"""

import os
import pathlib

import fire as fi
import polars as pl

from logger import get_logger
from vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    ref_version: str = "day_stays_qc",
    data_version: str = f"day_stays_qc_first_24h",
    data_dir: os.PathLike = "../clif-data/",
    icu_ids_loc: os.PathLike = "../mimiciv-3.1-icu-hids.parquet",
):
    data_dir, icu_ids_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (data_dir, icu_ids_loc),
    )

    # load and prep data
    splits = ("train", "val", "test")
    data_dirs = dict()
    ref_dirs = dict()
    for s in splits:
        data_dirs[s] = data_dir.joinpath(f"{data_version}-tokenized", s)
        ref_dirs[s] = data_dir.joinpath(f"{ref_version}-tokenized", s)

    vocab = Vocabulary().load(ref_dirs["train"].joinpath("vocab.gzip"))
    icu_stays = pl.scan_parquet(icu_ids_loc).with_columns(icu_stay=pl.lit(True))

    for s in splits:
        outcomes = (
            pl.scan_parquet(ref_dirs[s].joinpath("tokens_timelines.parquet"))
            .with_columns(
                length_of_stay=(
                    pl.col("times").list.get(-1) - pl.col("times").list.get(0)
                ).dt.total_hours(),
                same_admission_death=pl.col("tokens").list.contains(vocab("expired")),
            )
            .with_columns(
                long_length_of_stay=pl.col("length_of_stay") > 24 * 7  # 7 days in hours
            )
            .select(
                "hospitalization_id",
                "length_of_stay",
                "same_admission_death",
                "long_length_of_stay",
            )
            .join(
                icu_stays,
                how="left",
                on="hospitalization_id",
                validate="1:1",
                maintain_order="left",
            )
            .with_columns(pl.col("icu_stay").fill_null(False))
        )
        (
            pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines.parquet"))
            .join(
                outcomes,
                how="left",
                on="hospitalization_id",
                validate="1:1",
                maintain_order="left",
            )
            .collect()
            .write_parquet(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
        )


if __name__ == "__main__":
    fi.Fire(main)
