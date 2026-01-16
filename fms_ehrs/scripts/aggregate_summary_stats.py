#!/usr/bin/env python3

"""
generate summary statistics for cohorts
"""

import argparse
import pathlib

import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.tokenizer import Tokenizer21
from fms_ehrs.framework.tokenizer_base import summarize

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
    "--config_loc", type=pathlib.Path, default="../fms_ehrs/config/clif-21.yaml"
)
parser.add_argument("--data_version", type=str, default="V21")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

splits = ("train", "val", "test")
data_dirs = [pathlib.Path(dd).expanduser().resolve() for dd in args.data_dirs]

for data_dir in data_dirs:
    data_dirs = {
        s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits
    }
    tkzr = Tokenizer21(
        data_dir=data_dirs["train"],
        vocab_path=data_dirs["train"].joinpath("vocab.gzip"),
        config_file=args.config_loc,
    )

    def summarize_split(s):
        logger.info(f"split {s=}")
        tt = pl.read_parquet(data_dirs[s] / "tokens_timelines.parquet")
        n_hospitalizations = len(tt)
        n_tokens = tt.select(pl.col("tokens").list.len().sum()).item()
        logger.info(f"{n_hospitalizations=}")
        logger.info(f"{n_tokens=}")
        logger.info(f"avg_len={n_tokens / n_hospitalizations:.2f}")
        summarize(tokenizer=tkzr, tokens_timelines=tt, logger=logger)

    for s in splits:
        summarize_split(s)

tt_all = pl.concat(
    [
        pl.read_parquet(
            data_dir / f"{args.data_version}-tokenized" / s / "tokens_timelines.parquet"
        ).with_columns(split=pl.lit(s), set=pl.lit(data_dir.stem))
        for s in splits
        for data_dir in data_dirs
    ]
)
summarize(tokenizer=tkzr, tokens_timelines=tt_all, logger=logger)


logger.info("---fin")
