#!/usr/bin/env python3

"""
load different versions of tokenized timelines and generate descriptive statistics
"""

import argparse
import pathlib

import pandas as pd
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.tokenizer import Tokenizer21

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dirs",
    nargs="*",
    type=pathlib.Path,
    default=["../../data-mimic", "../../data-ucmc"],
)
parser.add_argument(
    "--data_versions", nargs="*", type=str, default=["Y21", "Y21_unfused"]
)
parser.add_argument(
    "--outcomes",
    nargs="*",
    type=str,
    default=[
        "icu_admission_24h",
        "imv_event_24h",
        "same_admission_death",
        "long_length_of_stay",
        "icu_admission",
        "imv_event",
        "ama_discharge",
        "hospice_discharge",
    ],
)
parser.add_argument("--splits", nargs="*", type=str, default=["train", "val", "test"])
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dirs = list(map(lambda d: pathlib.Path(d).expanduser().resolve(), args.data_dirs))

res = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [args.data_versions, (d.name.split("-")[-1] for d in data_dirs), args.splits],
        names=["version", "dataset", "split"],
    ),
    columns=["examples", "tot_tokens", "unq_tokens"],
)

for vers in args.data_versions:
    logger.info(vers.ljust(42, "="))
    for d in data_dirs:
        logger.info(d.name.split("-")[-1].upper().ljust(42, "-"))
        tkzr = Tokenizer21(
            data_dir=d / f"{vers}-tokenized" / "train",
            vocab_path=d / f"{vers}-tokenized" / "train" / "vocab.gzip",
            config_file=d / f"{vers}-tokenized" / "train" / "config.yaml",
        )
        logger.info(f"Vocab size: {len(tkzr.vocab)}")
        logger.info(f"{sorted(map(str, tkzr.vocab.lookup.keys()))=}")
        for s in args.splits:
            logger.info(f"{s} split")
            df = pl.scan_parquet(
                d / f"{vers}-tokenized" / s / "tokens_timelines.parquet"
            )
            n_ex = len(df.collect())
            n_tot = df.select("seq_len").sum().collect().item()
            n_unq = df.select(pl.col("tokens").explode().n_unique()).collect().item()
            res.loc[(vers, d.name.split("-")[-1], s)] = (n_ex, n_tot, n_unq)
            logger.info("top 20 tokens by usage:")
            logger.info(
                df.select(
                    pl.col("tokens")
                    .explode()
                    .replace_strict(tkzr.vocab.reverse, return_dtype=pl.String)
                )
                .group_by("tokens")
                .len()
                .sort("len", descending=True)
                .head(20)
                .collect()
            )
            logger.info(
                df.select(
                    pl.col("tokens")
                    .explode()
                    .replace_strict(
                        {
                            k: tkzr.get_token_type(v)
                            for k, v in tkzr.vocab.reverse.items()
                        },
                        return_dtype=pl.String,
                    )
                    .alias("token_type")
                )
                .group_by("token_type")
                .len()
                .sort("len", descending=True)
                .collect()
            )

logger.info(res)
logger.info(res.groupby(level=["version", "dataset"]).sum())


# vers = "Y21_first_24h"
# for agg in ("sum", "mean"):
#     with pl.Config(
#         tbl_rows=-1, tbl_cols=-1, tbl_width_chars=300, set_float_precision=3
#     ):
#         print(
#             pl.concat(
#                 pl.scan_parquet(
#                     d / f"{vers}-tokenized" / s / "tokens_timelines_outcomes.parquet"
#                 ).with_columns(split=pl.lit(s), dataset=pl.lit(d.name.split("-")[-1]))
#                 for s in args.splits
#                 for d in data_dirs
#             )
#             .group_by("dataset", "split")
#             .agg([getattr(pl.col(c), agg)() for c in args.outcomes])
#             .sort(
#                 ["dataset", pl.col("split").replace({"train": 0, "val": 1, "test": 2})]
#             )
#             .collect()
#         )

logger.info("---fin")
