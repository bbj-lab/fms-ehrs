#!/usr/bin/env python3

"""
Extract trajectory-related metrics for all data splits
"""

import argparse
import gzip
import pathlib

import numpy as np
import tqdm
from joblib import Parallel, delayed

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=pathlib.Path, default="/scratch/burkh4rt/data-ucmc"
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="V21")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-4476655-hp-V21",
)
parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
parser.add_argument("--all_layers", action="store_true")
parser.add_argument("--n_jobs", type=int, default=-1)

args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir),
)


def diff0(arr: np.ndarray, axis=0):
    return np.concatenate(
        [np.take(arr, indices=[0], axis=axis), np.diff(arr, axis=axis)], axis=axis
    )


for s in args.splits:
    featfiles = sorted(
        (data_dir / f"{args.data_version}-tokenized" / s).glob(
            "all-features{x}-{m}-batch*.npy.gz".format(
                x="-all-layers" if args.all_layers else "", m=model_loc.stem
            )
        ),
        key=lambda s: int(s.stem.strip(".npy").split("-batch")[-1]),
    )

    """jumps
    """

    get_jumps_from_shard = lambda f: np.linalg.norm(
        diff0(np.load(gzip.open(f, "rb")), axis=1), axis=2
    ).astype(np.float16)  # np.load(f) will have shape n_batch × tl_len × d_rep
    # or n_batch × tl_len × d_rep × (num_hidden_layers + 1) if args.all_layers

    jumps = np.concatenate(
        Parallel(n_jobs=args.n_jobs, verbose=True)(
            delayed(get_jumps_from_shard)(f)
            for f in tqdm.tqdm(featfiles, desc="shards")
        )
    )  # shape n_obs × tl_len or n_obs × tl_len × (num_hidden_layers + 1) if args.all_layers

    set_perms(np.save, compress=True)(
        out_dir
        / f"{args.data_version}-tokenized"
        / s
        / "all-jumps{x}-{m}.npy.gz".format(
            x="-all-layers" if args.all_layers else "", m=model_loc.stem
        ),
        jumps,
    )


logger.info("---fin")
