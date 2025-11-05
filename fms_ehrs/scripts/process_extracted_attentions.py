#!/usr/bin/env python3

"""
process importance metrics for all timelines in batches
"""

import argparse
import gzip
import itertools
import os
import pathlib
import typing

import numpy as np
import tqdm as tq

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike

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
parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
parser.add_argument(
    "--metrics",
    nargs="*",
    default=[
        "h2o-mean",
        "h2o-mean_log",
        "h2o-va-mean",
        "h2o-va-mean_log",
        "scissorhands-10",
        "scissorhands-20",
        "scissorhands-va-10",
        "scissorhands-va-20",
        "rollout-mean",
        "rollout-mean_log",
        "h2o-normed-mean",
        "h2o-normed-mean_log",
    ],
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

data_dirs = {
    s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in args.splits
}

for s in args.splits:
    for met in args.metrics:
        logger.info(f"{s=},{met=}")
        arrs = list(
            itertools.chain(
                data_dirs[s].glob(
                    "importance-{met}-{mdl}?*.npy".format(met=met, mdl=model_loc.stem)
                ),
                data_dirs[s].glob(
                    "importance-{met}-{mdl}?*.npy.gz".format(
                        met=met, mdl=model_loc.stem
                    )
                ),
            )
        )
        if len(arrs) < 2:
            logger.warning(f"For {s=} and {met=}, {arrs=}")
            logger.warning("Skipping...")
            continue
        with gzip.open(arrs.pop(), "rb") as f:
            arr = np.load(f).astype(np.float16)
        for arr_next in tq.tqdm(arrs):
            with gzip.open(arr_next, "rb") as f:
                new_arr = np.load(f).astype(np.float16)
                arr[new_arr != 0] = new_arr[new_arr != 0]
        if (mask := (arr == 0).all(axis=1)).any():
            logger.warning(
                f"Likely issues for {met=} in {s=} with {mask.sum()} arrays: {np.nonzero(mask)[0]}"
            )
        set_perms(np.save, compress=True)(
            data_dirs[s].joinpath(
                "importance-{met}-{mdl}.npy.gz".format(met=met, mdl=model_loc.stem)
            ),
            arr,
        )


logger.info("---fin")
