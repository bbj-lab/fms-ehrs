#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.util import collate_events_info, count_top_q

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama-med-60358922_1-hp-W++",
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc),
)

info = np.load(
    data_dir.joinpath(
        f"{args.data_version}-tokenized",
        "test",
        "log_probs-{m}.npy".format(m=model_loc.stem),
    )
) / -np.log(2)

tto = pl.read_parquet(
    data_dir.joinpath(
        f"{args.data_version}-tokenized",
        "test",
        "tokens_timelines_outcomes.parquet",
    )
)

assert info.shape[0] == tto.shape[0]

info_l = info.tolist()
times_l = tto.select("times").to_series().to_list()
info_agg_list = []
perp_agg_list = []

for tms, inf in zip(times_l, info_l):
    tlen = min(len(inf), len(tms))
    tms, inf = np.array(tms[:tlen]), np.array(inf[:tlen])
    info_agg_list.append(collate_events_info(tms, inf, aggregation="sum")[0])
    perp_agg_list.append(collate_events_info(tms, inf, aggregation="perplexity")[0])


tks_q99 = np.nansum(info >= np.nanquantile(info, q=0.99), axis=1)
tks_q95 = np.nansum(info >= np.nanquantile(info, q=0.95), axis=1)
tks_q95_99 = tks_q95 - tks_q99

evs_q99 = np.array(count_top_q(info_agg_list, q=0.99))
evs_q95 = np.array(count_top_q(info_agg_list, q=0.95))
evs_q95_99 = evs_q95 - evs_q99

outcomes = (
    "same_admission_death",
    "long_length_of_stay",
    "icu_admission",
    "icu_admission_24h",
    "imv_event",
    "imv_event_24h",
)
res = dict()
for outcome in outcomes:
    res[outcome] = tto.select(outcome).to_numpy().ravel().astype(int)

res["same_admission_death_24h"] = np.zeros_like(res["same_admission_death"])
res["long_length_of_stay_24h"] = np.zeros_like(res["same_admission_death"])

df = pd.DataFrame.from_dict(
    {
        "tks_q95": tks_q95,
        "evs_q99": evs_q99,
        "tks_q95_99": tks_q95_99,
        "evs_q95_99": evs_q95_99,
    }
    | res
)

lr = dict()
for outcome in outcomes:
    if not outcome.endswith("_24h"):
        logger.info(outcome)
        lr[outcome] = smf.logit(
            f"{outcome} ~ 1 + tks_q95 + evs_q95_99 + evs_q99",
            data=df.loc[lambda x: x[outcome + "_24h"] == 0],
        ).fit()
        logger.info(lr[outcome].summary())
        # logger.info(lr[outcome].summary().as_latex())

logger.info("---fin")
