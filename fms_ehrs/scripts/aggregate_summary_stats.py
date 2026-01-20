#!/usr/bin/env python3

"""
generate summary statistics for cohorts
"""

import argparse
import gzip
import pathlib

import numpy as np
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
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-4476655-hp-V21",
)
parser.add_argument("--data_version", type=str, default="V21")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

model_loc = pathlib.Path(args.model_loc).expanduser().resolve()
splits = ("train", "val", "test")
data_dirs = [pathlib.Path(dd).expanduser().resolve() for dd in args.data_dirs]

for data_dir in data_dirs[:1]:
    dds = {s: data_dir / f"{args.data_version}-tokenized" / s for s in splits}
    tkzr = Tokenizer21(
        data_dir=dds["train"],
        vocab_path=dds["train"] / "vocab.gzip",
        config_file=args.config_loc,
    )

    def summarize_split(s):
        logger.info(f"split {s=}")
        tt = pl.read_parquet(dds[s] / "tokens_timelines.parquet")
        n_hospitalizations = len(tt)
        n_tokens = tt.select(pl.col("tokens").list.len().sum()).item()
        logger.info(f"{n_hospitalizations=}")
        logger.info(f"{n_tokens=}")
        logger.info(f"avg_len={n_tokens / n_hospitalizations:.2f}")
        summarize(tokenizer=tkzr, tokens_timelines=tt, logger=logger)

    for s in splits:
        summarize_split(s)

tkns = np.vstack(
    pl.scan_parquet(dds["test"] / "tokens_timelines.parquet")
    .select("padded")
    .collect()
    .to_series()
    .to_numpy()
)
infm = np.load(
    gzip.open(dds["test"] / "information-{mdl}.npy.gz".format(mdl=model_loc.stem), "rb")
)
impt = np.load(
    gzip.open(
        dds["test"] / "importance-h2o-mean-{mdl}.npy.gz".format(mdl=model_loc.stem),
        "rb",
    )
)
df = pl.DataFrame(
    {
        "tkns": np.vectorize(tkzr.vocab.reverse.__getitem__)(tkns.ravel()),
        "typs": np.vectorize(tkzr.get_token_type_from_int)(tkns.ravel()),
        "infm": infm.ravel(),
        "impt": impt.ravel(),
        "ords": np.indices(tkns.shape)[1].ravel(),
    }
).filter(pl.col("tkns") != "PAD")

with pl.Config(tbl_rows=-1, set_float_precision=3):
    df.group_by("typs").agg(pl.col("infm").mean(), pl.col("impt").mean()).sort(
        "impt", descending=True
    )

with pl.Config(tbl_rows=-1, set_float_precision=3):
    df.group_by("tkns").agg(pl.col("infm").mean(), pl.col("impt").mean()).sort(
        "impt", descending=True
    ).head(25)

with pl.Config(tbl_rows=-1, set_float_precision=3):
    df.group_by("ords").agg(
        pl.col("infm").mean(),
        pl.col("impt").mean(),
        (pl.len() / len(tkns)).alias("freq"),
    ).sort("ords").head(100)


# tt_all = pl.concat(
#     [
#         pl.read_parquet(
#             data_dir / f"{args.data_version}-tokenized" / s / "tokens_timelines.parquet"
#         ).with_columns(split=pl.lit(s), set=pl.lit(data_dir.stem))
#         for s in splits
#         for data_dir in data_dirs
#     ]
# )
# summarize(tokenizer=tkzr, tokens_timelines=tt_all, logger=logger)


logger.info("---fin")
