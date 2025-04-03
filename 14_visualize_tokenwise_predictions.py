#!/usr/bin/env python3

"""
process results from 15_sft_predictions_over_time.py
"""

import argparse
import pathlib
import pickle

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from logger import get_logger
from util import mvg_avg as ma
from vocabulary import Vocabulary

pio.kaleido.scope.mathjax = None

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path)
parser.add_argument("--data_version", type=str)
parser.add_argument("--out_dir", type=pathlib.Path)
parser.add_argument("--model_loc", type=pathlib.Path)
args, unknowns = parser.parse_known_args()

model_loc, data_dir, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.model_loc, args.data_dir, args.out_dir),
)

data_version = args.data_version


# load and prep data
rng = np.random.default_rng(42)
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{data_version}-tokenized", s) for s in splits}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))


# open and unpack data

with open(
    data_dirs["test"].joinpath(f"sft_preds_tokenwise-{model_loc.stem}.pkl"),
    "rb",
) as fp:
    results = pickle.load(fp)


for mavg in (False, True):
    fig = go.Figure()

    for mt in results["mort_preds"].values():
        fig.add_trace(
            go.Scatter(
                y=ma(mt) if mavg else mt,
                mode="lines",
                opacity=0.2,
                line=dict(color="red"),
                name="Dies",
            )
        )

    for lt in results["live_preds"].values():
        fig.add_trace(
            go.Scatter(
                y=ma(lt) if mavg else lt,
                mode="lines",
                opacity=0.2,
                line=dict(color="blue"),
                name="Lives",
            )
        )

    for i in range(1, len(fig.data) - 1):
        fig.data[i].showlegend = False

    fig.update_layout(
        title="Predicted probability of death vs. number of tokens processed"
        + (" (smoothed)" if mavg else ""),
        xaxis_title="# tokens",
        yaxis_title="Predicted admission mortality prob."
        + (" (smoothed)" if mavg else ""),
    )

    # fig.show()
    fig.write_html(
        out_dir.joinpath(
            "tokenwise_vis-{m}{s}.html".format(
                m=model_loc.stem, s="-smooth" if mavg else ""
            )
        )
    )
    fig.write_image(
        out_dir.joinpath(
            "tokenwise_vis-{m}{s}.pdf".format(
                m=model_loc.stem, s="-smooth" if mavg else ""
            )
        )
    )
