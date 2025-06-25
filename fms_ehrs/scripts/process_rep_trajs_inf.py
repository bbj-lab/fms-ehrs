#!/usr/bin/env python3

"""
Do highly informative tokens correspond to bigger jumps in representation space?
"""

import argparse
import collections
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_pats
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import colors, plot_histogram, plot_histograms
from fms_ehrs.framework.tokenizer import token_type, standard_types
from fms_ehrs.framework.vocabulary import Vocabulary

mpl.rcParams["font.family"] = "cmr10"
mpl.rcParams["axes.formatter.use_mathtext"] = True

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
parser.add_argument(
    "--aggregation", choices=["sum", "max", "perplexity"], default="sum"
)
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
vocab = Vocabulary().load(
    data_dir.joinpath(f"{args.data_version}-tokenized", "train", "vocab.gzip")
)

jumps = np.load(test_dir.joinpath(f"all-jumps-{model_loc.stem}.npy"))
inf_arr = np.load(test_dir.joinpath(f"log_probs-{model_loc.stem}.npy")) / -np.log(2)
tt = pl.scan_parquet(test_dir.joinpath("tokens_timelines_outcomes.parquet"))
tks_arr = tt.select("padded").collect().to_series().to_numpy()
tms_arr = tt.select("times").collect().to_series().to_numpy()

assert jumps.shape == inf_arr[:, 1:].shape

"""
token-wise
"""

df_t = (
    pd.DataFrame(
        {
            "jump_length": jumps.ravel(),
            "information": inf_arr[:, 1:].ravel(),
            "token": np.row_stack(tks_arr)[:, 1:].ravel(),
        }
    )
    .assign(type=lambda df: df.token.map(lambda w: token_type(vocab.reverse[w])))
    .dropna()
)

logger.info(f"Tokenwise associations for {len(df_t)} tokens...")

lm_t = smf.ols(f"jump_length ~ 1 + information", data=df_t).fit()
logger.info(lm_t.summary())

if args.make_plots:
    colorer = dict(zip(standard_types, colors[1:]))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        df_t["information"],
        df_t["jump_length"],
        s=1,
        c=df_t.type.map(colorer.__getitem__).values,
        alpha=0.5,
    )

    legend_elements = [
        mpl_pats.Patch(facecolor=colorer[t], edgecolor="none", label=t)
        for t in standard_types
    ]
    ax.legend(handles=legend_elements, title="Type", loc="lower right")

    ax.set_title("Jump length vs. Information (Tokenwise)")
    ax.set_xlabel("information")
    ax.set_ylabel("jump_length")
    plt.savefig(
        out_dir.joinpath(
            "tokens-jumps-vs-infm-{m}-{d}.png".format(m=model_loc.stem, d=data_dir.stem)
        ),
        bbox_inches="tight",
        dpi=600,
    )

    plot_histogram(
        df_t["information"].values,
        savepath=out_dir.joinpath(
            "tokens-infm-{m}-{d}.png".format(m=model_loc.stem, d=data_dir.stem)
        ),
    )

    inf_by_type = collections.OrderedDict()
    for t in standard_types:
        inf_by_type[t] = df_t.loc[lambda df: df.type == t, "information"].values
    plot_histograms(
        inf_by_type,
        savepath=out_dir.joinpath(
            "tokens-infm-by-type-{m}-{d}.png".format(m=model_loc.stem, d=data_dir.stem)
        ),
    )

"""
event-wise
"""

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
    elif args.aggregation in ("sum", "perplexity"):
        event_info = np.zeros(shape=tms_unq.shape)
        np.add.at(event_info, idx, inf_i)
        # equivalent to:
        # event_info = np.bincount(idx, weights=inf_i.ravel(), minlength=tms_unq.shape[0])
        if args.aggregation == "perplexity":
            event_info /= np.bincount(idx, minlength=tms_unq.shape[0])
            np.exp2(event_info, out=event_info)
    else:
        raise Exception(f"Check {args.aggregation=}")
    event_jumps_sq = np.zeros(shape=tms_unq.shape)
    np.add.at(event_jumps_sq, idx, jsq_i)
    event_jumps = np.sqrt(event_jumps_sq)
    # equivalent to:
    # event_jumps = np.sqrt(
    #         np.bincount(idx, weights=jsq_i.ravel(), minlength=tms_unq.shape[0])
    #     )
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
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df_e["information"], df_e["jump_length"], s=1, c=colors[1], alpha=0.5)

    ax.set_title("Jump length vs. Information (Eventwise)")
    ax.set_xlabel("information")
    ax.set_ylabel("jump_length")
    plt.savefig(
        out_dir.joinpath(
            "events-jumps-vs-infm-{agg}-{m}-{d}.png".format(
                agg=args.aggregation, m=model_loc.stem, d=data_dir.stem
            )
        ),
        bbox_inches="tight",
        dpi=600,
    )

    plot_histogram(
        df_e["information"].values,
        savepath=out_dir.joinpath(
            "events-infm-{agg}-{m}-{d}.png".format(
                agg=args.aggregation, m=model_loc.stem, d=data_dir.stem
            )
        ),
    )

logger.info("---fin")
