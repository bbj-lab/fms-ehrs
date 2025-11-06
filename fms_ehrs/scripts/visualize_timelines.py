#!/usr/bin/env python3

"""
highlight timeline(s) according to provided metric(s)
"""

import argparse
import pathlib
import gzip

import numpy as np
from plotly import io as pio
import polars as pl
import tqdm as tq

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import imshow_text
from fms_ehrs.framework.storage import fix_perms
from fms_ehrs.framework.vocabulary import Vocabulary

try:
    pio.defaults.mathjax = None
except AttributeError:
    pio.kaleido.scope.mathjax = None

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="W++")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument(
    "--ids",
    type=str,
    nargs="*",
    default=[
        "24640534",  # cf. Fig. 2
        "26886976",  # Fig. 3
        "29022625",  # Fig. 4
    ],
)
parser.add_argument(
    "--metrics",
    type=str,
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
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--tl_len", type=int, default=210)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir),
)
out_dir.mkdir(parents=True, exist_ok=True)
fix_perms(out_dir)

# load and prep data
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

tto = (
    pl.scan_parquet(data_dirs["test"].joinpath("tokens_timelines.parquet"))
    .with_row_index()
    .filter(pl.col("hospitalization_id").is_in(args.ids))
    .with_columns(
        decoded=pl.col("tokens").list.eval(
            pl.element().map_elements(vocab.reverse.__getitem__, return_dtype=pl.String)
        )
    )
    .collect()
)

mets = {
    met: np.load(
        gzip.open(
            data_dirs["test"].joinpath(
                (
                    "importance-{met}-{mdl}.npy.gz"
                    if met.startswith(("h2o", "scissorhands", "rollout"))
                    else "saliency-{mdl}.npy.gz"
                ).format(met=met, mdl=model_loc.stem)
            ),
            "rb",
        )
    )[tto.select("index").to_numpy().ravel()]
    for met in args.metrics
}

n_cols = 6
n_rows = args.tl_len // n_cols
max_len = n_rows * n_cols
height = (700 * n_rows) // 42

for i, hid in tq.tqdm(
    enumerate(ids_list := tto.select("hospitalization_id").to_series().to_list()),
    total=len(ids_list),
):
    tl = np.vectorize(lambda s: s if len(s) <= 23 else f"{s[:13]}..{s[-7:]}")(
        tto.filter(pl.col("hospitalization_id") == hid)
        .select("decoded")
        .item()
        .to_numpy()
    )
    for k, v in mets.items():
        imshow_text(
            values=v[i, :max_len].reshape((-1, n_cols)),
            text=tl[:max_len].reshape((-1, n_cols)),
            savepath=out_dir.joinpath(
                "tls-{sid}-{met}-{dv}-{mv}.pdf".format(
                    sid=hid, met=k, dv=args.data_version, mv=model_loc.stem
                )
            ),
            autosize=False,
            zmin=v[i, :max_len].min(),
            zmax=v[i, :max_len].max(),
            height=height,
            width=1000,
            margin=dict(l=0, r=0, t=0, b=0),
        )


logger.info("---fin")
