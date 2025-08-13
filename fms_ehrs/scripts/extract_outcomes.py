#!/usr/bin/env python3

"""
determine some outcomes of interest for each hospitalization
"""

import os
import pathlib

import fire as fi
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    ref_version: str = "day_stays_qc",
    data_version: str = "day_stays_first_24h",
    data_dir: os.PathLike = None,
):
    data_dir = pathlib.Path(data_dir).expanduser().resolve()

    # load and prep data
    splits = ("train", "val", "test")
    data_dirs = dict()
    ref_dirs = dict()
    for s in splits:
        data_dirs[s] = data_dir.joinpath(f"{data_version}-tokenized", s)
        ref_dirs[s] = data_dir.joinpath(f"{ref_version}-tokenized", s)

    vocab = Vocabulary().load(ref_dirs["train"].joinpath("vocab.gzip"))
    expired_token = (
        vocab("expired") if "expired" in vocab.lookup else vocab("DSCG_expired")
    )
    icu_token = vocab("icu") if "icu" in vocab.lookup else vocab("ADT_icu")
    imv_token = vocab("imv") if "imv" in vocab.lookup else vocab("RESP_devc_imv")

    for s in splits:
        outcomes = (
            pl.scan_parquet(ref_dirs[s].joinpath("tokens_timelines.parquet"))
            .with_columns(
                length_of_stay=(
                    pl.col("times").list.get(-1) - pl.col("times").list.get(0)
                ).dt.total_hours(),
                same_admission_death=pl.col("tokens").list.contains(expired_token),
                icu_admission=pl.col("tokens").list.contains(icu_token),
                imv_event=pl.col("tokens").list.contains(imv_token),
            )
            .with_columns(
                long_length_of_stay=pl.col("length_of_stay") > 24 * 7  # 7 days in hours
            )
            .select(
                "hospitalization_id",
                "length_of_stay",
                "same_admission_death",
                "long_length_of_stay",
                "icu_admission",
                "imv_event",
            )
        )
        (
            pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines.parquet"))
            .with_columns(
                icu_admission_24h=pl.col("tokens").list.contains(icu_token),
                imv_event_24h=pl.col("tokens").list.contains(imv_token),
            )
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
