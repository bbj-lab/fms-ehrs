#!/usr/bin/env python3

"""
process results from sft_predictions_over_time
"""

import argparse
import pathlib
import pickle

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.util import ragged_lists_to_array

try:
    pio.defaults.mathjax = None
except AttributeError:
    pio.kaleido.scope.mathjax = None

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path)
parser.add_argument("--data_version", type=str)
parser.add_argument("--out_dir", type=pathlib.Path)
parser.add_argument("--model_loc", type=pathlib.Path)
parser.add_argument("--n", type=int, default=100)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

model_loc, data_dir, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.model_loc, args.data_dir, args.out_dir),
)

# load and prep data
rng = np.random.default_rng(42)
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits}


# open and unpack data

with open(
    data_dirs["test"].joinpath(
        "sft_preds-n" + str(args.n) + f"_tokenwise-{model_loc.stem}_LR.pkl"
    ),
    "rb",
) as fp:
    results = pickle.load(fp)


for suffix in ("", "_urt", "_lr"):
    fig = go.Figure()

    Mt = ragged_lists_to_array(results["mort_preds" + suffix].values())
    Mt_mean = np.nanmean(Mt, axis=0)
    Mt_2σ_hi = np.nanquantile(Mt, q=0.5 + 0.95 / 2, axis=0)
    Mt_2σ_lo = np.nanquantile(Mt, q=0.5 - 0.95 / 2, axis=0)

    fig.add_trace(
        go.Scatter(
            y=Mt_mean, mode="lines", line=dict(color="red", width=1), name="Dies"
        )
    )

    for i, y in enumerate((Mt_2σ_lo, Mt_2σ_hi)):
        fig.add_trace(
            go.Scatter(
                y=y,
                mode="lines",
                opacity=0.1,
                fill="tonexty" if i > 0 else None,
                line=dict(color="red", width=0),
                name="+/- 2σ",
            )
        )

    fig.data[1].showlegend = False

    Lt = ragged_lists_to_array(results["live_preds" + suffix].values())
    Lt_mean = np.nanmean(Lt, axis=0)
    Lt_2σ_hi = np.nanquantile(Lt, q=0.5 + 0.95 / 2, axis=0)
    Lt_2σ_lo = np.nanquantile(Lt, q=0.5 - 0.95 / 2, axis=0)

    fig.add_trace(
        go.Scatter(
            y=Lt_mean, mode="lines", line=dict(color="blue", width=1), name="Lives"
        )
    )

    for i, y in enumerate((Lt_2σ_lo, Lt_2σ_hi)):
        fig.add_trace(
            go.Scatter(
                y=y,
                mode="lines",
                opacity=0.1,
                fill="tonexty" if i > 0 else None,
                line=dict(color="blue", width=0),
                name="+/- 2σ",
            )
        )

    fig.data[4].showlegend = False

    fig.update_layout(
        title="Predicted probability of death vs. number of tokens processed "
        + suffix[1:].upper(),
        xaxis_title="# tokens",
        yaxis_title="Predicted admission mortality prob.",
        font_family="CMU Serif, Times New Roman, serif",
    )

    set_perms(fig.write_image)(
        out_dir.joinpath(
            "tokenwise_vis-{m}{s}-{d}.pdf".format(
                m=model_loc.stem, s=suffix, d=data_dir.stem
            )
        )
        .expanduser()
        .resolve()
    )


logger.info("---fin")
