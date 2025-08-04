#!/usr/bin/env python3

"""
Do highly informative events correspond to bigger jumps in representation space?
"""

import argparse
import itertools
import pathlib

import joblib as jl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
import torch as t
import tqdm

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import colors
from fms_ehrs.framework.util import collate_events_info

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
parser.add_argument(
    "--data_dir", type=pathlib.Path, default="/scratch/burkh4rt/data-mimic"
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument(
    "--aggregation", choices=["sum", "max", "perplexity"], default="sum"
)
parser.add_argument("--big_batch_sz", type=int, default=2**12)
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

inf_arr = np.load(test_dir.joinpath(f"log_probs-{model_loc.stem}.npy")) / -np.log(2)
if (f := test_dir.joinpath("tokens_timelines_outcomes.parquet")).exists():
    tt = pl.scan_parquet(f)
elif (f := test_dir.joinpath("tokens_timelines.parquet")).exists():
    tt = pl.scan_parquet(f)
else:
    raise FileNotFoundError("Check tokens_timelines* file.")
tks_arr = tt.select("padded").collect().to_series().to_numpy()
tms_arr = tt.select("times").collect().to_series().to_numpy()
ids = tt.select("hospitalization_id").collect().to_series().to_numpy()

featfiles = sorted(
    test_dir.glob("all-features-{m}-batch*.npy".format(m=model_loc.stem)),
    key=lambda s: int(s.stem.split("batch")[-1]),
)


def process_big_batch(batch_num: int, big_batch: t.Tensor):
    batch_reps = np.load(featfiles[batch_num]).astype(np.float16)
    batch_jumps = []
    batch_info = []
    for batch_idx, orig_idx in enumerate(big_batch.numpy()):
        tks, tms = tks_arr[orig_idx], tms_arr[orig_idx]
        tlen = min(
            len(tks), len(tms), np.argwhere(np.isfinite(inf_arr[orig_idx])).max() + 1
        )
        tks, tms = tks[:tlen], tms[:tlen]
        inf_i = np.nan_to_num(inf_arr[orig_idx, :tlen])
        event_info, time_idx = collate_events_info(tms, inf_i, args.aggregation)
        reps_i = batch_reps[batch_idx, :tlen]
        i_jumps = []
        i_info = []
        for e_i in pd.unique(time_idx)[1:]:  # preserves order, skips prefix
            aw = np.argwhere(time_idx == e_i)
            i_jumps.append(
                np.linalg.norm(reps_i[aw.max()] - reps_i[aw.min() - 1])
                .astype(np.float64)
                .item()
            )
            i_info.append(event_info[e_i].astype(np.float64).item())
        batch_jumps.append(i_jumps)
        batch_info.append(i_info)
    return batch_jumps, batch_info


batches_jumps_infos = jl.Parallel(n_jobs=-1, verbose=True)(
    jl.delayed(process_big_batch)(bn, bb)
    for bn, bb in tqdm.tqdm(
        enumerate(t.split(t.arange(len(tks_arr)), args.big_batch_sz))
    )
)

jumps = list(
    itertools.chain.from_iterable(jumps_infos[0] for jumps_infos in batches_jumps_infos)
)
infos = list(
    itertools.chain.from_iterable(jumps_infos[1] for jumps_infos in batches_jumps_infos)
)

assert len(jumps) == len(infos) == len(tks_arr)

tt_events = tt.with_columns(
    event_infos=pl.Series(infos), event_jumps=pl.Series(jumps)
).collect()
tt_events.write_parquet((test_dir.joinpath("tokens_timelines_outcomes_events.parquet")))

df_e = pd.DataFrame(
    {"total_jump": itertools.chain(*jumps), "information": itertools.chain(*infos)}
).dropna()

logger.info(f"Eventwise associations for {len(df_e)} events...")
lm_e = smf.ols(f"total_jump ~ 1 + information", data=df_e).fit()
logger.info(lm_e.summary())

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df_e["information"], df_e["total_jump"], s=1, c=colors[1], alpha=0.5)

ax.set_title("Total jump vs. Information (Eventwise)")
ax.set_xlabel("information")
ax.set_ylabel("total_jump")
plt.savefig(
    out_dir.joinpath(
        "jumps-vs-infm-{agg}-{m}-{d}.png".format(
            agg=args.aggregation, m=model_loc.stem, d=data_dir.stem
        )
    ),
    bbox_inches="tight",
    dpi=600,
)

## for highly informative events, >= 90th quantile
q = np.quantile(df_e.information, 0.95)
df_he = df_e.loc[lambda df: df.information >= q]

logger.info(f"Eventwise associations for {len(df_he)} informative events...")
lm_he = smf.ols("total_jump ~ 1 + information", data=df_he).fit()
logger.info(lm_he.summary())

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df_he["information"], df_he["total_jump"], s=1, c=colors[1], alpha=0.5)

ax.set_title("Total jump vs. Information (Informative events)")
ax.set_xlabel("information")
ax.set_ylabel("total_jump")
plt.savefig(
    out_dir.joinpath(
        "jumps-vs-infm-{agg}-{m}-{d}-he.png".format(
            agg=args.aggregation, m=model_loc.stem, d=data_dir.stem
        )
    ),
    bbox_inches="tight",
    dpi=600,
)

logger.info("---fin")
