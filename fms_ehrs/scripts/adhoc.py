#!/usr/bin/env python3

"""
for a list of models, collect predictions and compare performance
"""

import argparse
import pathlib

import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.tokenizer import Tokenizer21

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="~/Downloads/data-mimic")
parser.add_argument("--data_version", type=str, default="Y21")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.out_dir)
)

# load and prep data
splits = ("train", "val", "test")
data_dirs = {s: data_dir / f"{args.data_version}-tokenized" / s for s in splits}


tt = pl.scan_parquet(data_dirs["train"] / "tokens_timelines.parquet")

tkzr = Tokenizer21(
    data_dir=data_dirs["train"],
    vocab_path=data_dirs["train"] / "vocab.gzip",
    config_file=data_dirs["train"] / "config.yaml",
)

with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=100):
    print(
        tt.select(
            pl.col("tokens")
            .explode()
            .replace_strict(tkzr.vocab.reverse, return_dtype=pl.String)
        )
        .with_columns(
            pairs=pl.concat_str(
                [pl.col("tokens"), pl.col("tokens").shift(-1)], separator=","
            )
        )
        .filter(pl.col("tokens") != "TL_END")
        .group_by("pairs")
        .len()
        .sort("len", descending=True)
        .head(100)
        .collect()
    )

with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=100):
    print(
        tt.select(
            pl.col("tokens")
            .explode()
            .replace_strict(tkzr.vocab.reverse, return_dtype=pl.String)
        )
        .with_columns(
            triplets=pl.concat_str(
                [
                    pl.col("tokens"),
                    pl.col("tokens").shift(-1),
                    pl.col("tokens").shift(-2),
                ],
                separator=",",
            )
        )
        .filter(pl.col("tokens") != "TL_END")
        .filter(pl.col("tokens").shift(-1) != "TL_END")
        .group_by("triplets")
        .len()
        .sort("len", descending=True)
        .head(100)
        .collect()
    )

with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=100):
    print(
        tt.select(
            pl.col("tokens")
            .explode()
            .replace_strict(tkzr.vocab.reverse, return_dtype=pl.String)
        )
        .with_columns(
            quadruples=pl.concat_str(
                [
                    pl.col("tokens"),
                    pl.col("tokens").shift(-1),
                    pl.col("tokens").shift(-2),
                    pl.col("tokens").shift(-3),
                ],
                separator=",",
            )
        )
        .filter(pl.col("tokens") != "TL_END")
        .filter(pl.col("tokens").shift(-1) != "TL_END")
        .filter(pl.col("tokens").shift(-2) != "TL_END")
        .group_by("quadruples")
        .len()
        .sort("len", descending=True)
        .head(100)
        .collect()
    )


# with pl.Config(tbl_rows=-1):
#     print(rass.group_by("numerical_value").len().sort("numerical_value"))
