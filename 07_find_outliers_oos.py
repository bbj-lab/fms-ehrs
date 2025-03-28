#!/usr/bin/env python3

"""
- determine if outliers @24h correlate with adverse events
"""

import argparse
import pathlib

import numpy as np
import polars as pl
import sklearn.ensemble as skl_ens

from logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser(
    description="Determine outliers for both in-sample and out-of-sample datasets."
)
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../clif-data")
parser.add_argument(
    "--data_dir_new", type=pathlib.Path, default="/scratch/burkh4rt/clif-data"
)
parser.add_argument("--data_version", type=str, default="day_stays_qc_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630",
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc, args.out_dir),
)
data_version = args.data_version


def summarize(mort: np.array, llos: np.array, out: np.array):
    for outcome in ("mortality", "long_los"):
        logger.info(outcome)
        for s in splits:
            logger.info(s)
            xtab = (
                pl.DataFrame(
                    {
                        "mortality": mort[s],
                        "long_los": llos[s],
                        "outlier_24h": out[s] == -1,
                    }
                )
                .group_by([outcome, "outlier_24h"])
                .agg(pl.len().alias("count"))
                .pivot(
                    values="count", index="outlier_24h", on=outcome, sort_columns=True
                )
            )
            logger.info(xtab)

            xtab_in = xtab.row(by_predicate=~pl.col("outlier_24h"), named=True)
            in_outcome_rate = xtab_in["true"] / (xtab_in["true"] + xtab_in["false"])
            logger.info(
                "inlier {o} rate: {r:.2f}%".format(o=outcome, r=100 * in_outcome_rate)
            )

            xtab_out = xtab.row(by_predicate=pl.col("outlier_24h"), named=True)
            out_outcome_rate = xtab_out["true"] / (xtab_out["true"] + xtab_out["false"])
            logger.info(
                "outlier {o} rate: {r:.2f}%".format(o=outcome, r=100 * out_outcome_rate)
            )

            logger.info(
                "risk factor: {:.2f}".format(out_outcome_rate / in_outcome_rate)
            )


"""
run on original data
"""

logger.info(f"Running in {data_dir_orig}")

splits = ("train", "val", "test")
data_dirs = dict()
feats = dict()
mort = dict()
llos = dict()
for s in splits:
    data_dirs[s] = data_dir_orig.joinpath(f"{data_version}-tokenized", s)
    feats[s] = np.load(
        data_dirs[s].joinpath("features-{m}.npy".format(m=model_loc.stem))
    )
    mort[s] = (
        pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .ravel()
    )
    llos[s] = (
        pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
        .select("long_length_of_stay")
        .collect()
        .to_numpy()
        .ravel()
    )

for s in splits:
    assert feats[s].shape[0] == mort[s].shape[0]


clf = skl_ens.IsolationForest(
    random_state=42
)  # "Returns -1 for outliers and 1 for inliers"
out = dict()
out["train"] = clf.fit_predict(feats["train"])
logger.info(
    "train: {n} ({pct:.2f}%) outliers in {ntot}".format(
        n=(out["train"] == -1).sum(),
        pct=100 * (out["train"] == -1).mean(),
        ntot=out["train"].size,
    )
)
for s in ("val", "test"):
    out[s] = clf.predict(feats[s])
    logger.info(
        "{s}: {n} ({pct:.2f}%) outliers in {ntot}".format(
            s=s,
            n=(out[s] == -1).sum(),
            pct=100 * (out[s] == -1).mean(),
            ntot=out[s].size,
        )
    )
for s in splits:
    np.save(
        data_dirs[s].joinpath("features-outliers-{m}.npy".format(m=model_loc.stem)),
        out[s],
    )

summarize(mort, llos, out)

"""
run on new data
"""

logger.info(f"Running in {data_dir_new}")

data_dirs = dict()
feats = dict()
mort = dict()
llos = dict()
for s in splits:
    data_dirs[s] = data_dir_new.joinpath(f"{data_version}-tokenized", s)
    feats[s] = np.load(
        data_dirs[s].joinpath("features-{m}.npy".format(m=model_loc.stem))
    )
    mort[s] = (
        pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .ravel()
    )
    llos[s] = (
        pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
        .select("long_length_of_stay")
        .collect()
        .to_numpy()
        .ravel()
    )

for s in splits:
    assert feats[s].shape[0] == mort[s].shape[0]


for s in splits:
    out[s] = clf.predict(feats[s])
    logger.info(
        "{s}: {n} ({pct:.2f}%) outliers in {ntot}".format(
            s=s,
            n=(out[s] == -1).sum(),
            pct=100 * (out[s] == -1).mean(),
            ntot=out[s].size,
        )
    )

summarize(mort, llos, out)

for s in splits:
    np.save(
        data_dirs[s].joinpath("features-outliers-{m}.npy".format(m=model_loc.stem)),
        out[s],
    )
