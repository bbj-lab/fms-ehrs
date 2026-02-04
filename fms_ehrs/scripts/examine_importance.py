#!/usr/bin/env python3

"""
Examine results from gathered token importances
"""

import argparse
import gzip
import pathlib

import numpy as np
import polars as pl
import seaborn as sns

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.tokenizer_old import token_type
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--data_version", type=str, default="W++")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
parser.add_argument(
    "--metrics",
    nargs="*",
    default=[
        "h2o-mean",
        "h2o-mean_log",
        "h2o-va-mean",
        "h2o-va-mean_log",
        "scissorhands-10",
        "scissorhands-20",
        "scissorhands-va-10",
        "scissorhands-va-20",
        "rollout-mean",
        "rollout-mean_log",
        "h2o-normed-mean",
        "h2o-normed-mean_log",
    ],
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.out_dir, args.model_loc),
)

data_dirs = {s: data_dir / f"{args.data_version}-tokenized" / s for s in args.splits}

vocab = Vocabulary().load(
    data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip"
)
pad_tkn = vocab("PAD")

splits = ("train", "val", "test")
df = pl.concat(
    [
        pl.read_parquet(data_dirs[s] / "tokens_timelines.parquet").with_columns(
            split=pl.lit(s)
        )
        for s in args.splits
    ]
)

infm = np.concatenate(
    [
        np.load(data_dirs[s] / "log_probs-{m}.npy".format(m=model_loc.stem))
        for s in args.splits
    ]
) / -np.log(2)

metrics = {
    met: np.concatenate(
        [
            np.load(
                gzip.open(
                    data_dirs[s]
                    / "importance-{met}-{mdl}.npy.gz".format(
                        met=met, mdl=model_loc.stem
                    ),
                    "rb",
                )
            )
            for s in args.splits
        ]
    )
    for met in args.metrics
}

ravelled = (
    pl.from_dict(
        {k: v.ravel() for k, v in metrics.items()}
        | {"tk_id": np.stack(df["padded"].to_list()).ravel()}
        | {"information": infm.ravel()}
    )
    .filter(~pl.any_horizontal(pl.all().is_nan()))
    .with_columns(
        tk=pl.col("tk_id").map_elements(
            vocab.reverse.__getitem__, return_dtype=pl.String
        )
    )
    .with_columns(tk_typ=pl.col("tk").map_elements(token_type, return_dtype=pl.String))
    .drop("tk_id")
)

p = sns.pairplot(ravelled.sample(n_samp := 10_000).to_pandas(), hue="tk_typ")
set_perms(p.figure.savefig)(
    out_dir
    / "importance-metrics-samp{ns}-{m}-{d}.pdf".format(
        ns=n_samp, m=model_loc.stem, d=data_dir.stem
    ),
    bbox_inches="tight",
)

logger.info("Average importance by token type")
with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2):
    logger.info(
        ravelled.drop("tk")
        .group_by("tk_typ")
        .mean()
        # .select("tk_typ", "h2o-mean", "h2o-va-mean", "rollout-mean", "h2o-normed-mean")
        .sort("h2o-normed-mean")
    )
