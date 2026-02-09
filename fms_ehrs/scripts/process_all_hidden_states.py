#!/usr/bin/env python3

"""
Extract trajectory-related metrics for all data splits
"""

import argparse
import gzip
import pathlib
import pickle

import numpy as np
import sklvq
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
parser.add_argument("--gmlvq_dir", type=pathlib.Path, default="../../data-mimic")
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

data_dir, model_loc, out_dir, gmlvq_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir, args.gmlvq_dir),
)

with open(
    gmlvq_dir
    / (args.data_version + "-tokenized")
    / "train"
    / ("gmlvq-" + model_loc.stem + ".pkl"),
    "rb",
) as fp:
    pkl = pickle.load(fp)
    scaler = pkl["scaler"]
    model = pkl["model"]


def diff0(arr: np.ndarray, axis=0):
    return np.concatenate(
        [np.take(arr, indices=[0], axis=axis), np.diff(arr, axis=axis)], axis=axis
    )


def get_importances_1(X: np.ndarray, category: str):
    Xt = X[: np.argmin(isf.all(axis=1))] if not (isf := np.isfinite(X)).all() else X
    Xt = scaler.transform(Xt)  # apply PCA
    class_proto_idx = np.char.find(model.classes_.astype(str), category) > -1
    closest_class_proto = np.argmin(
        sklvq.distances.AdaptiveSquaredEuclidean()(Xt, model)[:, class_proto_idx],
        axis=1,
    )
    class_protos_m = model.transform(model.get_prototypes()[class_proto_idx])
    X_m = model.transform(Xt)
    X_m1 = np.zeros_like(X_m)
    X_m1[1:] = X_m[:-1]
    dX_m = diff0(X_m)
    abs_importances = np.zeros(shape=(len(X_m), len(class_protos_m)))
    rel_importances = np.zeros(shape=(len(X_m), len(class_protos_m)))
    for i, p in enumerate(class_protos_m):
        abs_importances[:, i] = np.dot(dX_m, p) / np.linalg.norm(p)
        rel_importances[:, i] = np.einsum("ij,ij->i", dX_m, p - X_m1) / np.linalg.norm(
            p - X_m1, axis=1
        )
    abs_importances = abs_importances[np.arange(len(X_m)), closest_class_proto]
    rel_importances = rel_importances[np.arange(len(X_m)), closest_class_proto]
    return abs_importances, rel_importances


def get_importances_all(X: np.ndarray, category: str):
    abs_importances_all = np.zeros_like(X[:, :, 0])
    rel_importances_all = np.zeros_like(X[:, :, 0])
    for i, x in enumerate(X):
        abs_importances, rel_importances = get_importances_1(x, category)
        abs_importances_all[i, : len(abs_importances)] = abs_importances
        rel_importances_all[i, : len(rel_importances)] = rel_importances
    return abs_importances_all, rel_importances_all


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
    for ai, ri in Parallel(n_jobs=args.n_jobs, verbose=True, return_as="generator")(
        delayed(process_shard)(f, category) for f in tqdm.tqdm(featfiles, desc="shards")
    ):
        ais.append(ai)
        ris.append(ri)

    set_perms(np.save, compress=True)(
        out_dir
        / f"{args.data_version}-tokenized"
        / "test"
        / "abs-imp-{c}-{m}.npy.gz".format(c=category, m=model_loc.stem),
        np.concatenate(ais),
    )

    set_perms(np.save, compress=True)(
        out_dir
        / f"{args.data_version}-tokenized"
        / "test"
        / "rel-imp-{c}-{m}.npy.gz".format(c=category, m=model_loc.stem),
        np.concatenate(ris),
    )


for outcome in args.outcomes:
    logger.info(f"Processing {outcome}...")
    run_category(outcome)

logger.info("---fin")
