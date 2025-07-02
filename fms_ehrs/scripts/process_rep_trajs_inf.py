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
import seaborn as sns
import statsmodels.formula.api as smf

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import colors, plot_histogram, plot_histograms
from fms_ehrs.framework.tokenizer import token_type, token_types
from fms_ehrs.framework.util import collate_events_info
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
parser.add_argument("--skip_kde", action="store_true")
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
colorer = dict(zip(token_types, colors[1:]))

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
    .loc[lambda df: (df.jump_length > 0) & (df.information > 0)]
)

logger.info(f"Tokenwise associations for {len(df_t)} tokens...")

lm_t = smf.ols(f"jump_length ~ 1 + information", data=df_t).fit()
logger.info(lm_t.summary())

if args.make_plots:

    plot_histogram(
        df_t["information"].values,
        savepath=out_dir.joinpath(
            "tokens-infm-{m}-{d}.png".format(m=model_loc.stem, d=data_dir.stem)
        ),
    )

    inf_by_type = collections.OrderedDict()
    for t in token_types:
        inf_by_type[t] = df_t.loc[lambda df: df.type == t, "information"].values
    plot_histograms(
        inf_by_type,
        savepath=out_dir.joinpath(
            "tokens-infm-by-type-{m}-{d}.png".format(m=model_loc.stem, d=data_dir.stem)
        ),
    )

if not args.skip_kde:
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.kdeplot(
        data=df_t,
        x="information",
        y="jump_length",
        hue="type",
        levels=1,
        thresh=0.001,
        ax=ax,
        palette=colorer,
    )

    legend_elements = [
        mpl_pats.Patch(facecolor=colorer[t], edgecolor="none", label=t)
        for t in token_types
    ]
    ax.legend(handles=legend_elements, title="Type", loc="lower right")

    ax.set_title("Jump length vs. Information (Tokenwise)")
    ax.set_xlabel("information")
    ax.set_ylabel("jump_length")
    plt.savefig(
        out_dir.joinpath(
            "tokens-jumps-vs-infm-{m}-{d}.svg".format(m=model_loc.stem, d=data_dir.stem)
        ),
        bbox_inches="tight",
    )

"""
event-wise
"""

jumps_padded = np.column_stack([np.zeros(jumps.shape[0]), jumps])

info_list = list()
path_len_list = list()
event_len_list = list()

for i in range(len(tks_arr)):
    tks, tms = tks_arr[i], tms_arr[i]
    tlen = min(len(tks), len(tms))
    tks, tms = tks[:tlen], tms[:tlen]
    inf_i = np.nan_to_num(inf_arr[i, :tlen])
    j_i = np.nan_to_num(jumps_padded[i, :tlen])
    event_info, idx = collate_events_info(tms, inf_i, args.aggregation)
    path_lens = np.bincount(idx, weights=j_i.ravel(), minlength=event_info.shape[0])
    event_lens = np.bincount(idx, minlength=event_info.shape[0])
    if args.drop_prefix:
        event_info = np.delete(event_info, idx[0])
        path_lens = np.delete(path_lens, idx[0])
        event_lens = np.delete(event_lens, idx[0])
    info_list += event_info.tolist()
    path_len_list += path_lens.tolist()
    event_len_list += event_lens.tolist()

assert len(info_list) == len(path_len_list) == len(event_len_list)

df_e = pd.DataFrame(
    {
        "path_length": path_len_list,
        "information": info_list,
        "event_length": event_len_list,
    }
).dropna()

logger.info(f"Eventwise associations for {len(df_e)} events...")
lm_e = smf.ols(f"path_length ~ 1 + information", data=df_e).fit()
logger.info(lm_e.summary())

lm_el = smf.ols(f"information ~ 1 + event_length", data=df_e).fit()
logger.info(lm_el.summary())

if args.make_plots:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df_e["information"], df_e["path_length"], s=1, c=colors[1], alpha=0.5)

    ax.set_title("Path length vs. Information (Eventwise)")
    ax.set_xlabel("information")
    ax.set_ylabel("path_length")
    plt.savefig(
        out_dir.joinpath(
            "path-lens-vs-infm-{agg}-{m}-{d}.png".format(
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

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df_e["event_length"], df_e["information"], s=1, c=colors[1], alpha=0.5)

    ax.set_title("Information vs. event length")
    ax.set_xlabel("event_length")
    ax.set_ylabel("information")
    plt.savefig(
        out_dir.joinpath(
            "infm-vs-event-len-{agg}-{m}-{d}.png".format(
                agg=args.aggregation, m=model_loc.stem, d=data_dir.stem
            )
        ),
        bbox_inches="tight",
        dpi=600,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df_e["event_length"], df_e["path_length"], s=1, c=colors[1], alpha=0.5)

    ax.set_title("Path length vs. event length")
    ax.set_xlabel("event_length")
    ax.set_ylabel("path_length")
    plt.savefig(
        out_dir.joinpath(
            "path-vs-event-len-{agg}-{m}-{d}.png".format(
                agg=args.aggregation, m=model_loc.stem, d=data_dir.stem
            )
        ),
        bbox_inches="tight",
        dpi=600,
    )

logger.info("---fin")
