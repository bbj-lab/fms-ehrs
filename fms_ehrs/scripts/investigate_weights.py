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

to_lookup = list()

for s in splits:
    print(s)
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
    print("total: {}".format(len(ids)))
    print("with token:")
    w_tk = tk_mask.any(axis=1)
    print("   fraction: {:>4.2f}".format(w_tk.mean()))
    print("   count: {:>7}".format(w_tk.sum()))
    d_tk = np.array(
        [
            max(vals[i], default=0) - min(vals[i], default=0)
            for i, m in enumerate(w_tk)
            if m
        ]
    )
    # print("of these, that move >=1 deciles:")
    # m_tk = d_tk >= 1
    # print("   fraction: {:>4.2f}".format(m_tk.mean()))
    # print("   count: {:>7}".format(m_tk.sum()))
    print("of these, that move >=2 deciles:")
    m_tk = d_tk >= 2
    print("   fraction: {:>4.2f}".format(m_tk.mean()))
    print("   count: {:>7}".format(m_tk.sum()))
    print("   worst cases (largest total variation):")
    tv_tk = np.array([np.abs(np.diff(vals[i])).sum() for i, m in enumerate(w_tk) if m])
    asort = np.argsort(tv_tk)[::-1][:10]
    for i in asort:
        print(
            "      {} with {}".format(ids[w_tk][i], vals[np.arange(len(vals))[w_tk][i]])
        )
        tt = np.array(
            [
                (
                    (d if len(d) <= 23 else f"{d[:13]}..{d[-7:]}")
                    if (d := vocab.reverse[t]) is not None
                    else "None"
                )
                for t in tl[w_tk][i]
            ]
        )
        imshow_text(
            values=np.nan_to_num(infm[w_tk][i])[:max_len].reshape((-1, n_cols)),
            text=tt[:max_len].reshape((-1, n_cols)),
            # title=f"Information by token for patient {s} in {names[v]}",
            savepath=out_dir.joinpath(
                "tokens-{v}-{s}-{m}-hist.pdf".format(
                    v=ids[w_tk][i], s=s, m=model_loc.stem
                )
            ),
            autosize=False,
            zmin=0,
            zmax=30,
            height=height,
            width=1000,
            margin=dict(l=0, r=0, t=0, b=0),
        )
    to_lookup += ids[w_tk][asort].tolist()
    surprisingly_the_same_score = np.array(
        [
            max(infs[i][1:]) if len(set(vals[i])) == 1 and len(vals[i]) > 1 else 0
            for i in range(len(tl))
        ]
    )
    asort = np.argsort(surprisingly_the_same_score)[::-1][:10]
    ls_k = ids[asort]
    print(
        "   most surprisingly-the-same cases (highest surprise on a repeat measurement):"
    )
    for i in asort:
        print(
            "      {} with {}\n".format(ids[i], vals[i]),
            "              scores: {}".format(infs[i].round(2)),
        )
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
    print("-" * 42)
    print(i)
    print("orig:")
    print(
        df_events.filter(pl.col("hadm_id") == int(i)).sort(
            pl.col("charttime").str.to_datetime()
        )
    )
    print("clif:")
    print(df_clif.filter(pl.col("hospitalization_id") == i).sort("recorded_dttm"))


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

"""
train
total: 291390
with token:
   fraction: 0.20
   count:   57285
of these, that move >=2 deciles:
   fraction: 0.05
   count:    2912
   worst cases (largest total variation):
      28812737 with [6 6 0 6 0 6 6]
      26262781 with [0 7 0 7]
      26528130 with [9 2 2 9 2]
      21581611 with [0 5 0 0 5 0 0]
      23494336 with [0 4 0 4 0 1 0 1]
      20820288 with [0 6 0 0 0 7]
      29339233 with [9 0 0 9 9]
      28910506 with [9 0 9 9 9]
      26367525 with [0 0 9 0]
      23408678 with [9 9 0 0 9 9 9]
   most surprisingly-the-same cases (highest surprise on a repeat measurement):
      23619860 with [1 1 1 1 1 1 1]
               scores: [18.05 22.66 27.27 22.5  26.1  23.96 17.22]
      20606203 with [0 0 0 0 0 0]
               scores: [18.28 27.11 17.06 17.55 20.48 18.92]
      20381058 with [9 9 9 9 9 9 9]
               scores: [19.31 19.59 18.34 19.7  26.95 20.79 23.19]
      29378082 with [9 9]
               scores: [20.45 26.8 ]
      26272149 with [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
               scores: [19.98 21.9  26.78 25.07 18.82 18.62 18.44 22.89 23.23 17.74 17.5  17.69
 23.88 17.86 18.67 17.98 24.58 23.48 22.61 25.02 23.56 17.54 24.51]
      29866426 with [0 0 0 0 0]
               scores: [18.33 21.89 18.91 17.19 26.78]
      28555350 with [0 0]
               scores: [17.25 26.77]
      22965393 with [0 0 0 0]
               scores: [18.13 26.72 17.83 17.57]
      22137973 with [0 0 0 0 0]
               scores: [19.28 26.62 17.77 25.35 19.16]
      25939873 with [0 0 0]
               scores: [18.77 26.61 17.56]
val
total: 43099
with token:
   fraction: 0.19
   count:    8259
of these, that move >=2 deciles:
   fraction: 0.05
   count:     436
   worst cases (largest total variation):
      26750128 with [9 0 9]
      29756945 with [9 0 9 9 9]
      27291415 with [8 0 8]
      20602578 with [8 0 8]
      22322462 with [8 0 8 8]
      27784119 with [9 1 9 9]
      20293891 with [9 1 1 9 9 9]
      29680091 with [9 1 9 9 9]
      21591002 with [1 9 1 1]
      26361537 with [7 7 0 0 7 8]
   most surprisingly-the-same cases (highest surprise on a repeat measurement):
      28822348 with [0 0 0 0 0]
               scores: [19.24 20.5  26.74 14.79 15.72]
      24021148 with [0 0 0 0 0 0 0 0]
               scores: [17.93 26.18 26.61 18.39 24.52 17.7  17.78 17.79]
      20383411 with [0 0 0]
               scores: [19.1  26.54 19.56]
      20507645 with [0 0 0 0 0]
               scores: [18.77 20.42 26.43 19.08 23.86]
      25160085 with [4 4 4 4]
               scores: [17.76 19.63 26.33 20.3 ]
      22299273 with [0 0 0 0 0 0 0]
               scores: [18.77 20.13 18.77 19.31 26.07 15.92 23.32]
      21382601 with [9 9]
               scores: [20.38 26.07]
      26914099 with [1 1 1 1 1 1]
               scores: [16.02 18.9  23.98 15.7  19.68 25.85]
      29099127 with [9 9 9 9 9 9 9 9 9 9 9 9]
               scores: [20.32 17.98 19.77 17.51 19.11 25.79 18.84 18.4  16.18 23.85 18.63 23.81]
      20843425 with [1 1 1]
               scores: [17.04 17.35 25.78]
test
total: 90816
with token:
   fraction: 0.18
   count:   16596
of these, that move >=2 deciles:
   fraction: 0.05
   count:     798
   worst cases (largest total variation):
      29694591 with [5 6 0 0 6 0 0 5]
      23427725 with [7 2 7 2 2 7 7]
      21747890 with [0 0 0 0 9 0]
      29385031 with [9 0 9 9]
      22328710 with [9 0 9 9]
      22535104 with [9 9 9 0 9]
      21350747 with [0 0 9 0]
      28833781 with [1 1 9 1 1]
      27826819 with [8 0 8 8 8]
      20772904 with [1 9 1]
   most surprisingly-the-same cases (highest surprise on a repeat measurement):
      28833314 with [0 0]
               scores: [18.07 27.28]
      24740974 with [0 0 0]
               scores: [17.65 26.58 14.89]
      23173235 with [0 0 0 0 0 0]
               scores: [17.7  26.56 15.19 18.07 18.01 24.25]
      27390600 with [2 2]
               scores: [18.89 26.47]
      29298288 with [0 0]
               scores: [19.2  26.34]
      21666785 with [0 0 0 0]
               scores: [17.7  19.3  26.24 15.62]
      25324232 with [2 2 2 2]
               scores: [18.42 26.23 18.72 17.81]
      22787930 with [0 0 0 0 0 0 0]
               scores: [21.57 26.18 18.63 16.8  16.83 23.96 17.73]
      27913245 with [0 0 0 0 0]
               scores: [17.95 18.25 17.3  26.13 24.63]
      24513877 with [0 0 0]
               scores: [17.41 21.13 26.09]
"""
