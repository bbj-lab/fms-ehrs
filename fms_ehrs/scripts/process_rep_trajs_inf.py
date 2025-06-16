#!/usr/bin/env python3

"""
Do highly informative tokens correspond to bigger jumps in representation space?
"""

import argparse
import pathlib

import bokeh.io as bk_io
import bokeh.plotting as bk_plt
import bokeh.models as bk_mdls
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

from fms_ehrs.framework.logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../clif-data-ucmc")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--aggregation", choices=["sum", "max"], default="sum")
parser.add_argument("--drop_prefix", action="store_true")
parser.add_argument("--make_plots", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.out_dir, args.model_loc),
)
test_dir = data_dir.joinpath(f"{args.data_version}-tokenized", "test")


"""
token-wise
"""

jumps = np.load(test_dir.joinpath(f"all-jumps-{model_loc.stem}.npy"))
inf_arr = np.load(test_dir.joinpath(f"log_probs-{model_loc.stem}.npy")) / -np.log(2)

assert jumps.shape == inf_arr[:, 1:].shape

df_t = pd.DataFrame(
    {"jump_length": jumps.ravel(), "information": inf_arr[:, 1:].ravel()}
).dropna()

logger.info(f"Tokenwise associations for {len(df_t)} tokens...")

lm_t = smf.ols(f"jump_length ~ 1 + information", data=df_t).fit()
logger.info(lm_t.summary())

if args.make_plots:
    p = bk_plt.figure(
        title="Jump length vs. Information (Event‑wise)",
        x_axis_label="information",
        y_axis_label="jump_length",
        width=600,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.scatter(
        x="information", y="jump_length", size=2, source=bk_mdls.ColumnDataSource(df_t)
    )

    p.title.text_font = "Computer Modern Serif"
    p.xaxis.axis_label_text_font = p.yaxis.axis_label_text_font = (
        "Computer Modern Serif"
    )
    p.background_fill_color = "white"
    bk_io.save(
        p,
        filename=out_dir.joinpath(
            "tokens-jumps-vs-infm-{m}-{d}.html".format(
                m=model_loc.stem, d=data_dir.stem
            )
        ),
    )

"""
event-wise
"""

tt = pl.scan_parquet(test_dir.joinpath("tokens_timelines_outcomes.parquet"))
tks_arr = tt.select("padded").collect().to_series().to_numpy()
tms_arr = tt.select("times").collect().to_series().to_numpy()
jsq = np.column_stack([np.zeros(jumps.shape[0]), np.square(jumps)])

info_list = list()
jump_list = list()

for i in range(len(tks_arr)):
    tks, tms = tks_arr[i], tms_arr[i]
    tlen = min(len(tks), len(tms))
    tks, tms = tks[:tlen], tms[:tlen]
    tms_unq, idx = np.unique(tms, return_inverse=True)
    inf_i = np.nan_to_num(inf_arr[i, :tlen])
    jsq_i = np.nan_to_num(jsq[i, :tlen])
    if args.aggregation == "max":
        event_info = np.full(tms_unq.shape, -np.inf)
        np.maximum.at(event_info, idx, inf_i)
    elif args.aggregation == "sum":
        event_info = np.zeros(shape=tms_unq.shape)
        np.add.at(event_info, idx, inf_i)
    else:
        raise Exception(f"Check {args.aggregation=}")
    event_jumps_sq = np.zeros(shape=tms_unq.shape)
    np.add.at(event_jumps_sq, idx, jsq_i)
    event_jumps = np.sqrt(event_jumps_sq)
    if args.drop_prefix:
        event_info = np.delete(event_info, idx[0])
        event_jumps = np.delete(event_jumps, idx[0])
    info_list += event_info.tolist()
    jump_list += event_jumps.tolist()

assert len(info_list) == len(jump_list)

df_e = pd.DataFrame({"jump_length": jump_list, "information": info_list}).dropna()

logger.info(f"Eventwise associations for {len(df_e)} tokens...")
lm_e = smf.ols(f"jump_length ~ 1 + information", data=df_e).fit()
logger.info(lm_e.summary())

if args.make_plots:
    p = bk_plt.figure(
        title="Jump length vs. Information (Event‑wise)",
        x_axis_label="information",
        y_axis_label="jump_length",
        width=600,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.scatter(
        x="information", y="jump_length", size=2, source=bk_mdls.ColumnDataSource(df_e)
    )

    p.title.text_font = "Computer Modern Serif"
    p.xaxis.axis_label_text_font = p.yaxis.axis_label_text_font = (
        "Computer Modern Serif"
    )
    p.background_fill_color = "white"
    bk_io.save(
        p,
        filename=out_dir.joinpath(
            "events-jumps-vs-infm-{m}-{d}.html".format(
                m=model_loc.stem, d=data_dir.stem
            )
        ),
    )

logger.info("---fin")
