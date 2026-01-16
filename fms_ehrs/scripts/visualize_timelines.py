#!/usr/bin/env python3

"""
highlight timeline(s) according to provided metric(s)
"""

import argparse
import gzip
import pathlib

import numpy as np
import polars as pl
import tqdm as tq
from plotly import io as pio

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
parser.add_argument("--data_version", type=str, default="V21")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-4476655-hp-V21",
)
parser.add_argument(
    "--ids",
    type=str,
    nargs="*",
    default=[
        "27416443",
        "29161299",
        "28311451",
        "23636379",
        "21294017",
        "27323869",
        "24116026",
        "25523459",
        "29835765",
        "27854643",
        "21837764",
        "29761794",
    ],
)
parser.add_argument(
    "--metrics",
    type=str,
    nargs="*",
    default=[
        "h2o-mean",
        # "h2o-mean_log",
        # "h2o-va-mean",
        # "h2o-va-mean_log",
        # "scissorhands-10",
        # "scissorhands-20",
        # "scissorhands-va-10",
        # "scissorhands-va-20",
        # "rollout-mean",
        # "rollout-mean_log",
        # "h2o-normed-mean",
        # "h2o-normed-mean_log",
        "information",
    ],
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--tl_len", type=int, default=300)
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

# lookup = dict(
#     enumerate(
#         pl.scan_parquet(data_dirs["test"].joinpath("tokens_timelines.parquet"))
#         .select("hospitalization_id")
#         .collect()
#         .to_series()
#         .to_list()
#     )
# )

# infm = np.load(
#     gzip.open(
#         data_dirs["test"].joinpath(
#             ("information-{mdl}.npy.gz").format(mdl=model_loc.stem)
#         ),
#         "rb",
#     )
# )[:, 7:300]

# [
#     lookup[i]
#     for i in np.argsort(np.max(infm, axis=1) * np.isfinite(infm[:, 299 - 7]))[::-1][:50]
# ]

# np.random.default_rng().choice(list(lookup.values()), size=50, replace=False).tolist()

tt = (
    pl.scan_parquet(data_dirs["test"].joinpath("tokens_timelines.parquet"))
    .with_row_index()
    .filter(pl.col("hospitalization_id").is_in(args.ids))
    .with_columns(
        decoded=pl.col("padded").list.eval(
            pl.element().map_elements(
                vocab.reverse.__getitem__, return_dtype=pl.String, skip_nulls=False
            )
        )
    )
    .collect()
)

mets = {
    met: np.load(
        gzip.open(
            data_dirs["test"].joinpath(
                (
                    "{met}-{mdl}.npy.gz"
                    if met.startswith("information")
                    else "importance-{met}-{mdl}.npy.gz"
                ).format(met=met, mdl=model_loc.stem)
            ),
            "rb",
        )
    )[tt.select("index").to_numpy().ravel()]
    for met in args.metrics
}

if data_dir.stem == "data-ucmc" and "information" in mets.keys():
    # manually fix issue with admission types
    mets["information"][:, 5] = 0

n_cols = 6
n_rows = args.tl_len // n_cols
max_len = n_rows * n_cols
height = (700 * n_rows) // 42

for i, hid in tq.tqdm(
    enumerate(ids_list := tt.select("hospitalization_id").to_series().to_list()),
    total=len(ids_list),
):
    tl = np.vectorize(
        lambda s: "None" if s is None else s if len(s) <= 23 else f"{s[:13]}..{s[-7:]}"
    )(
        tt.filter(pl.col("hospitalization_id") == hid)
        .select("decoded")
        .item()
        .to_numpy()
    )
    if (mis_len := max_len - len(tl)) > 0:
        tl = np.append(tl, mis_len * ["—"])
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
