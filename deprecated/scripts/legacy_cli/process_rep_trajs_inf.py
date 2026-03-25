#!/usr/bin/env python3

"""
Do highly informative tokens correspond to bigger jumps in representation space?
"""

import argparse
import pathlib

import matplotlib.patches as mpl_pats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import colors, plot_histogram
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.tokenizer import token_type, token_types, type_names
from fms_ehrs.framework.util import collate_events_info
from fms_ehrs.framework.vocabulary import Vocabulary

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts,microtype}",
    }
)

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
if (f := test_dir.joinpath("tokens_timelines_outcomes.parquet")).exists():
    tt = pl.scan_parquet(f)
elif (f := test_dir.joinpath("tokens_timelines.parquet")).exists():
    tt = pl.scan_parquet(f)
else:
    raise FileNotFoundError("Check tokens_timelines* file.")
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
    .assign(
        type=lambda df: df.token.map(lambda w: type_names[token_type(vocab.reverse[w])])
    )
    .dropna()
    .loc[lambda df: (df.jump_length > 0) & (df.information > 0)]
)

logger.info(f"Tokenwise associations for {len(df_t)} tokens...")

lm_t = smf.ols("jump_length ~ 1 + information", data=df_t).fit()
logger.info(lm_t.summary())

if args.make_plots:
    fig = px.scatter(
        df_t.groupby("type").agg("mean").reset_index(),
        x="information",
        y="jump_length",
        color="type",
        symbol="type",
        color_discrete_sequence=colors[1:],
    )
    fig.update_layout(
        # title="Average jump magnitude vs. average information by token type",
        xaxis_title="Information (bits)",
        yaxis_title="Jump Magnitude (in representation space)",
        template="plotly_white",
        font_family="CMU Serif, Times New Roman, serif",
    )
    set_perms(fig.write_image)(
        out_dir.joinpath(
            "twise-jumps-infs-{m}-{d}.pdf".format(m=model_loc.stem, d=data_dir.stem)
        )
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
    set_perms(plt.savefig)(
        out_dir.joinpath(
            "tokens-jumps-vs-infm-{m}-{d}.pdf".format(m=model_loc.stem, d=data_dir.stem)
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
lm_e = smf.ols("path_length ~ 1 + information", data=df_e).fit()
logger.info(lm_e.summary())

lm_el = smf.ols("information ~ 1 + event_length", data=df_e).fit()
logger.info(lm_el.summary())

if args.make_plots:
    fig = px.scatter(
        df_e, x="information", y="path_length", color_discrete_sequence=colors[1:]
    )
    fig.update_traces(marker_size=1)
    fig.update_layout(
        # title="Path length vs. information by event",
        xaxis_title="Information (bits)",
        yaxis_title="Path length (in representation space)",
        template="plotly_white",
        font_family="CMU Serif, Times New Roman, serif",
    )
    set_perms(fig.write_image)(
        out_dir.joinpath(
            "path-lens-vs-infm-{agg}-{m}-{d}.pdf".format(
                agg=args.aggregation, m=model_loc.stem, d=data_dir.stem
            )
        )
    )

    plot_histogram(
        df_e["information"].values,
        savepath=out_dir.joinpath(
            "events-infm-{agg}-{m}-{d}.pdf".format(
                agg=args.aggregation, m=model_loc.stem, d=data_dir.stem
            )
        ),
    )


logger.info("---fin")
