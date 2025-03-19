#!/usr/bin/env python3

"""
process results from 15_sft_predictions_over_time.py
"""

import os
import pathlib
import pickle

import numpy as np
import plotly.graph_objects as go

from logger import get_logger
from vocabulary import Vocabulary
from util import mvg_avg as ma

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

model_dir: os.PathLike = "../clif-mdls-archive/mdl-llama1b-sft-57451707-clsfr"
data_dir: os.PathLike = "../clif-data/day_stays_qc_first_24h-tokenized"
out_dir: os.PathLike = "../"

model_dir, data_dir, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (model_dir, data_dir, out_dir),
)

# load and prep data
rng = np.random.default_rng(42)
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(s) for s in splits}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))


# open and unpack data

with open(
    data_dirs["test"].joinpath(f"sft_preds_tokenwise-{model_dir.stem}.pkl"),
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
                m=model_dir.stem, s="-smooth" if mavg else ""
            )
        )
    )
