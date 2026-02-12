#!/usr/bin/env python3

"""
Extract trajectory-related metrics for all data splits
"""

import argparse
import gzip
import pathlib
import pickle

import numpy as np
import scipy as sp
import tqdm
from joblib import Parallel, delayed

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=pathlib.Path, default="/scratch/burkh4rt/data-mimic"
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--proto_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="Y21_first_24h")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/gemma-5635921-Y21"
)
parser.add_argument(
    "--outcomes",
    nargs="+",
    default=[
        "same_admission_death",
        "long_length_of_stay",
        # "ama_discharge",
        # "hospice_discharge",
    ],
)
parser.add_argument("--n_jobs", type=int, default=-1)

args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir, proto_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir, args.proto_dir),
)

eps = np.finfo(float).eps

with open(
    proto_dir
    / (args.data_version + "-tokenized")
    / "train"
    / ("lda-gmm-protos-" + model_loc.stem + ".pkl"),
    "rb",
) as fp:
    pkl = pickle.load(fp)
    scaler = pkl["scaler"]
    models = pkl["models"]


proto_trees = {
    outcome: sp.spatial.cKDTree(models[outcome].means_) for outcome in args.outcomes
}


def diff0(arr: np.ndarray, axis=0):
    return np.concatenate(
        [np.take(arr, indices=[0], axis=axis), np.diff(arr, axis=axis)], axis=axis
    )


def get_importances_1(X: np.ndarray, category: str):
    Xt = X[: np.argmin(isf.all(axis=1))] if not (isf := np.isfinite(X)).all() else X
    Xt = scaler.transform(Xt)  # apply LDA
    class_protos = models[category].means_  # these were learned in the LDA space
    class_precisions = models[category].precisions_
    class_memberships = models[category].predict_proba(Xt)
    closest_class_proto = proto_trees[category].query(Xt)[-1]
    Xt1 = np.zeros_like(Xt)
    Xt1[1:] = Xt[:-1]
    dXt = diff0(Xt)
    abs_importances = np.zeros(shape=(len(Xt), len(class_protos)))
    rel_importances = np.zeros(shape=(len(Xt), len(class_protos)))
    abs_gmm = np.zeros(shape=(len(Xt), len(class_protos)))
    rel_gmm = np.zeros(shape=(len(Xt), len(class_protos)))
    for i in range(models[category].n_components):
        p = class_protos[i]
        abs_importances[:, i] = (dXt @ p) / (np.linalg.norm(p) + eps)
        rel_importances[:, i] = np.einsum("ij,ij->i", dXt, p - Xt1) / (
            np.linalg.norm(p - Xt1, axis=1) + eps
        )
        S = class_precisions[i]
        abs_gmm[:, i] = (dXt @ S @ p) / (np.sqrt(np.dot(p, S @ p)) + eps)
        rel_gmm[:, i] = np.einsum("ij,ij->i", dXt @ S, p - Xt1) / (
            np.sqrt(np.einsum("ij,ij->i", (p - Xt1) @ S, p - Xt1)) + eps
        )
    abs_importances = abs_importances[np.arange(len(Xt)), closest_class_proto]
    rel_importances = rel_importances[np.arange(len(Xt)), closest_class_proto]
    abs_gmm = np.sum(abs_gmm * class_memberships, axis=1)
    rel_gmm = np.sum(rel_gmm * class_memberships, axis=1)
    return abs_importances, rel_importances, abs_gmm, rel_gmm


def get_importances_all(X: np.ndarray, category: str):
    abs_importances_all = np.zeros_like(X[:, :, 0])
    rel_importances_all = np.zeros_like(X[:, :, 0])
    abs_gmm_all = np.zeros_like(X[:, :, 0])
    rel_gmm_all = np.zeros_like(X[:, :, 0])
    for i, x in enumerate(X):
        abs_importances, rel_importances, abs_gmm, rel_gmm = get_importances_1(
            x, category
        )
        abs_importances_all[i, : len(abs_importances)] = abs_importances
        rel_importances_all[i, : len(rel_importances)] = rel_importances
        abs_gmm_all[i, : len(abs_gmm)] = abs_gmm
        rel_gmm_all[i, : len(rel_gmm)] = rel_gmm
    return abs_importances_all, rel_importances_all, abs_gmm_all, rel_gmm_all


def process_shard(f, category):
    X = np.load(gzip.open(f, "rb"))
    return get_importances_all(X, category)


def run_category(category):
    featfiles = sorted(
        (data_dir / f"{args.data_version}-tokenized" / "test").glob(
            "all-features-{m}-batch*.npy.gz".format(m=model_loc.stem)
        ),
        key=lambda s: int(s.stem.strip(".npy").split("-batch")[-1]),
    )

    ais = []
    ris = []
    ais_gmm = []
    ris_gmm = []
    for ai, ri, ag, rg in Parallel(
        n_jobs=args.n_jobs, verbose=True, return_as="generator"
    )(delayed(process_shard)(f, category) for f in tqdm.tqdm(featfiles, desc="shards")):
        ais.append(ai)
        ris.append(ri)
        ais_gmm.append(ag)
        ris_gmm.append(rg)

    for met, lst in {
        "abs-imp": ais,
        "rel-imp": ris,
        "abs-gmm": ais_gmm,
        "rel-gmm": ris_gmm,
    }.items():
        set_perms(np.save, compress=True)(
            out_dir
            / f"{args.data_version}-tokenized"
            / "test"
            / "{met}-{c}-{m}.npy.gz".format(met=met, c=category, m=model_loc.stem),
            np.concatenate(lst),
        )


for outcome in args.outcomes:
    logger.info(f"Processing {outcome}...")
    run_category(outcome)

logger.info("---fin")
