#!/usr/bin/env python3

"""
- determine if outliers @24h correlate with adverse events
"""

import argparse
import pathlib

import numpy as np
import sklearn.ensemble as skl_ens

from src.framework.logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path)
parser.add_argument("--data_dir_new", type=pathlib.Path)
parser.add_argument("--data_version", type=str)
parser.add_argument("--model_loc", type=pathlib.Path)
parser.add_argument("--out_dir", type=pathlib.Path)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc, args.out_dir),
)
data_version = args.data_version

"""
run on original data
"""

logger.info(f"Running in {data_dir_orig}")

splits = ("train", "val", "test")
data_dirs = dict()
feats = dict()

for s in splits:
    data_dirs[s] = data_dir_orig.joinpath(f"{data_version}-tokenized", s)
    feats[s] = np.load(
        data_dirs[s].joinpath("features-{m}.npy".format(m=model_loc.stem))
    )

clf = skl_ens.IsolationForest(
    random_state=42
)  # "Returns -1 for outliers and 1 for inliers"
out = dict()
anom = dict()
out["train"] = clf.fit_predict(feats["train"])
logger.info(
    "train: {n} ({pct:.2f}%) outliers in {ntot}".format(
        n=(out["train"] == -1).sum(),
        pct=100 * (out["train"] == -1).mean(),
        ntot=out["train"].size,
    )
)
anom["train"] = -1.0 * clf.score_samples(
    feats["train"]
)  # "Opposite of the anomaly score defined in the original paper"
logger.info(
    "anomaly score: mean {mn:.2f}, std {sd:.2f}, median {md:.2f}".format(
        mn=anom["train"].mean(), sd=anom["train"].std(), md=np.median(anom["train"])
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
    anom[s] = -1.0 * clf.score_samples(feats[s])
    logger.info(
        "anomaly score: mean {mn:.2f}, std {sd:.2f}, median {md:.2f}".format(
            mn=anom[s].mean(), sd=anom[s].std(), md=np.median(anom[s])
        )
    )

for s in splits:
    np.save(
        data_dirs[s].joinpath("features-outliers-{m}.npy".format(m=model_loc.stem)),
        out[s],
    )
    np.save(
        data_dirs[s].joinpath(
            "features-anomaly-score-{m}.npy".format(m=model_loc.stem)
        ),
        anom[s],
    )

"""
run on new data
"""

logger.info(f"Running in {data_dir_new}")

data_dirs = dict()
feats = dict()

for s in splits:
    data_dirs[s] = data_dir_new.joinpath(f"{data_version}-tokenized", s)
    feats[s] = np.load(
        data_dirs[s].joinpath("features-{m}.npy".format(m=model_loc.stem))
    )
    out[s] = clf.predict(feats[s])
    logger.info(
        "{s}: {n} ({pct:.2f}%) outliers in {ntot}".format(
            s=s,
            n=(out[s] == -1).sum(),
            pct=100 * (out[s] == -1).mean(),
            ntot=out[s].size,
        )
    )

    anom[s] = -1.0 * clf.score_samples(feats[s])
    print(anom[s])
    logger.info(
        "anomaly score: mean {mn:.2f}, std {sd:.2f}, median {md:.2f}".format(
            mn=anom[s].mean(), sd=anom[s].std(), md=np.median(anom[s])
        )
    )

    np.save(
        data_dirs[s].joinpath("features-outliers-{m}.npy".format(m=model_loc.stem)),
        out[s],
    )
    np.save(
        data_dirs[s].joinpath(
            "features-anomaly-score-{m}.npy".format(m=model_loc.stem)
        ),
        anom[s],
    )

logger.info("---fin")
