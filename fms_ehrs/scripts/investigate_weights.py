#!/usr/bin/env python3

"""
investigate why repeated measurements of the same weight may appear surprising
(and why some weights fluctuate as much as they do)
"""

import argparse
import pathlib

import numpy as np
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import imshow_text
from fms_ehrs.framework.vocabulary import Vocabulary

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
parser.add_argument("--token", type=str, default="VTL_weight_kg")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--max_len", type=int, default=300)
parser.add_argument("--save_plots", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir),
)

splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits}
rng = np.random.default_rng(42)
vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
n_cols = 6
n_rows = args.max_len // n_cols
max_len = n_rows * n_cols
height = (700 * n_rows) // 42


def plot_tl(timeline, information, name):
    tt = np.array(
        [
            (
                (d if len(d) <= 23 else f"{d[:13]}..{d[-7:]}")
                if (d := vocab.reverse[t]) is not None
                else "None"
            )
            for t in timeline
        ]
    )
    imshow_text(
        values=np.nan_to_num(information)[:max_len].reshape((-1, n_cols)),
        text=tt[:max_len].reshape((-1, n_cols)),
        savepath=out_dir.joinpath(
            "tokens-{n}-{m}-hist.pdf".format(n=name, m=model_loc.stem)
        ),
        autosize=False,
        zmin=0,
        zmax=30,
        height=height,
        width=1000,
        margin=dict(l=0, r=0, t=0, b=0),
    )


to_lookup = list()
for s in splits:
    logger.info(s)
    tl = np.array(
        pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines.parquet"))
        .select("padded")
        .collect()
        .to_series()
        .to_list()
    )
    ids = np.array(
        pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines.parquet"))
        .select("hospitalization_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    infm = np.load(
        data_dirs[s].joinpath("log_probs-{m}.npy".format(m=model_loc.stem))
    ) / -np.log(2)
    tk_mask = tl == vocab.lookup[args.token]
    val_mask = np.logical_and(
        np.roll(tk_mask, 1, axis=1), np.logical_and(0 <= tl, tl < 10)
    )
    vals = [tl[i, val_mask[i]] for i in range(len(tl))]
    infs = [infm[i, val_mask[i]] for i in range(len(tl))]
    logger.info("total: {}".format(len(ids)))
    logger.info("with token:")
    w_tk = tk_mask.any(axis=1)
    logger.info("   fraction: {:>4.2f}".format(w_tk.mean()))
    logger.info("   count: {:>7}".format(w_tk.sum()))
    d_tk = np.array(
        [
            max(vals[i], default=0) - min(vals[i], default=0)
            for i, m in enumerate(w_tk)
            if m
        ]
    )
    # logger.info("of these, that move >=1 deciles:")
    # m_tk = d_tk >= 1
    # logger.info("   fraction: {:>4.2f}".format(m_tk.mean()))
    # logger.info("   count: {:>7}".format(m_tk.sum()))
    logger.info("of these, that move >=2 deciles:")
    m_tk = d_tk >= 2
    logger.info("   fraction: {:>4.2f}".format(m_tk.mean()))
    logger.info("   count: {:>7}".format(m_tk.sum()))
    logger.info("   worst cases (largest total variation):")
    tv_tk = np.array([np.abs(np.diff(vals[i])).sum() for i, m in enumerate(w_tk) if m])
    asort = np.argsort(tv_tk)[::-1][:10]
    for i in asort:
        logger.info(
            "      {} with {}".format(ids[w_tk][i], vals[np.arange(len(vals))[w_tk][i]])
        )
        if args.save_plots:
            plot_tl(timeline=tl[w_tk][i], information=infm[w_tk][i], name=ids[w_tk][i])
    to_lookup += ids[w_tk][asort].tolist()
    surprisingly_the_same_score = np.array(
        [
            max(infs[i][1:]) if len(set(vals[i])) == 1 and len(vals[i]) > 1 else 0
            for i in range(len(tl))
        ]
    )
    asort = np.argsort(surprisingly_the_same_score)[::-1][:10]
    ls_k = ids[asort]
    logger.info(
        "   most surprisingly-the-same cases (highest surprise on a repeat measurement):"
    )
    for i in asort:
        logger.info("      {} with {}".format(ids[i], vals[i]))
        logger.info("              scores: {}".format(infs[i].round(2)))
        if args.save_plots:
            plot_tl(timeline=tl[i], information=infm[i], name=ids[i])
    to_lookup += ids[asort].tolist()


df_events = (
    pl.scan_csv("/gpfs/data/bbj-lab/users/burkh4rt/mimiciv-3.1/icu/chartevents.csv.gz")
    .join(
        pl.scan_csv(
            "/gpfs/data/bbj-lab/users/burkh4rt/mimiciv-3.1/icu/d_items.csv.gz",
            infer_schema_length=100_000,
        ),
        on="itemid",
    )
    .filter(
        pl.col("hadm_id").is_in([int(i) for i in to_lookup])
        & pl.col("label").str.to_lowercase().str.contains("weight")
    )
    .select("hadm_id", "charttime", "value", "valuenum", "label", "abbreviation")
    .collect()
)

df_clif = (
    pl.scan_parquet(
        "/gpfs/data/bbj-lab/users/burkh4rt/CLIF-MIMICv0.2.0/output/rclif-2.1/clif_vitals.parquet"
    )
    .filter(
        pl.col("hospitalization_id").is_in(list(to_lookup))
        & pl.col("vital_name").str.to_lowercase().str.contains("weight")
    )
    .collect()
)

for i in to_lookup:
    logger.info("-" * 42)
    logger.info(i)
    logger.info("orig:")
    logger.info(
        df_events.filter(pl.col("hadm_id") == int(i)).sort(
            pl.col("charttime").str.to_datetime()
        )
    )
    logger.info("clif:")
    logger.info(df_clif.filter(pl.col("hospitalization_id") == i).sort("recorded_dttm"))


# df_omr = (
#     pl.scan_csv(
#         "/gpfs/data/bbj-lab/users/burkh4rt/mimiciv-3.1/hosp/omr.csv.gz",
#         infer_schema_length=100_000,
#     )
#     .filter(
#         pl.col("result_name").str.to_lowercase().str.contains("weight")
#         & pl.col("hadm_id").is_in([int(i) for i in to_lookup])
#     )
#     .sort(pl.col("chartdate").str.to_datetime())
#     .select("hadm_id", "chartdate", "result_name", "result_value")
#     .collect()
# )
