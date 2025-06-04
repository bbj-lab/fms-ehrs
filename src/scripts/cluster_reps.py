#!/usr/bin/env python3

"""
cluster 24h-representations
"""

import argparse
import pathlib
import pickle

import numpy as np
import sklearn as skl

from src.framework.logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../clif-data-ucmc")
parser.add_argument("--data_version", type=str, default="QC_noX_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama1b-original-59946215-hp-QC_noX",
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc),
)

versions = ("orig", "new")
data_dirs = {
    v: (data_dir_orig if v == "orig" else data_dir_new).joinpath(
        f"{args.data_version}-tokenized", "test"
    )
    for v in versions
}

for v in versions:
    features = np.load(
        data_dirs[v].joinpath("features-{m}.npy".format(m=model_loc.stem))
    )
    logger.info(f"{v}...")
    clusterer = skl.cluster.HDBSCAN(
        store_centers="medoid", min_cluster_size=100, n_jobs=-1
    )
    clusterer.fit(features)

    logger.info(f"Found {max(clusterer.labels_)+1} clusters.")

    with open(
        data_dirs[v].joinpath("dbscan-reps-{m}-m100.pkl".format(m=model_loc.stem)),
        "wb",
    ) as fp:
        pickle.dump(clusterer, fp)


logger.info("---fin")
