#!/usr/bin/env python3

"""
Extract trajectory-related metrics for all data splits
"""

import argparse
import pathlib

import numpy as np
import tqdm
from joblib import Parallel, delayed

from fms_ehrs.framework.logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/llama1b-57928921-run1"
)
parser.add_argument("splits", nargs="*", default=["train", "val", "test"])
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

for s in args.splits:
    featfiles = sorted(
        data_dir.joinpath(f"{args.data_version}-tokenized", s).glob(
            "all-features-{m}-batch*.npy".format(m=model_loc.stem)
        ),
        key=lambda x: int(x.stem.split("batch")[-1]),
    )

    """jumps
    """

    get_jumps_from_shard = lambda f: np.linalg.norm(
        np.diff(np.load(f), axis=1), axis=-1
    ).astype(np.float16)  # np.load(f) will have shape n_obs × tl_len × d_rep

    jumps = np.concatenate(
        Parallel(n_jobs=-1, verbose=True)(
            delayed(get_jumps_from_shard)(f)
            for f in tqdm.tqdm(featfiles, desc="shards")
        )
    )  # shape n_obs × tl_len-1

    np.save(
        data_dir.joinpath(
            f"{args.data_version}-tokenized",
            s,
            "all-jumps-{m}.npy".format(m=model_loc.stem),
        ),
        jumps,
    )

    """alignments
    """

    def get_alignments_from_shard(f: pathlib.Path) -> np.array:
        x = np.load(f)
        return np.einsum(
            "ijk,ijk->ij",
            x[:, 1:, :] - x[:, 0, :][:, np.newaxis, :],
            np.diff(x, axis=1),
        ) / np.arange(1, x.shape[1])

    alignments = np.concatenate(
        Parallel(n_jobs=-1, verbose=True)(
            delayed(get_alignments_from_shard)(f)
            for f in tqdm.tqdm(featfiles, desc="shards")
        )
    )  # shape n_obs × tl_len-1

    np.save(
        data_dir.joinpath(
            f"{args.data_version}-tokenized",
            s,
            "all-alignments-{m}.npy".format(m=model_loc.stem),
        ),
        alignments,
    )


logger.info("---fin")
