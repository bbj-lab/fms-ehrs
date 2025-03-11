#!/usr/bin/env python3

"""
Are trajectory summary statistics at 24h predictive of various outcomes?
"""

import os
import pathlib

import fire as fi
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
import tqdm

from logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    data_dir: os.PathLike = "../clif-data",
    data_version: str = "day_stays_qc_first_24h",
    model_loc: os.PathLike = "../clif-mdls-archive/mdl-day_stays_qc-llama1b-57350630",
):

    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (data_dir, model_loc),
    )

    featfiles = sorted(
        data_dir.joinpath(f"{data_version}-tokenized", "test").glob(
            "all-features-{m}-batch*.npy".format(m=model_loc.stem)
        ),
        key=lambda s: int(s.stem.split("batch")[-1]),
    )

    get_jumps = lambda x: np.linalg.norm(
        np.diff(x, axis=1), axis=-1
    )  # x will have shape n_obs × tl_len × d_rep

    traj_lens = []
    max_jumps = []
    avg_jumps = []

    for f in tqdm.tqdm(featfiles, desc="shards"):
        j = get_jumps(
            np.load(f).astype(np.float64)
        )  # rounding is ok but overflow is not
        traj_lens.append(np.nansum(j, axis=-1))
        max_jumps.append(np.nanmax(j, axis=-1))
        avg_jumps.append(np.nanmean(j, axis=-1))

    traj_len = np.concatenate(traj_lens)
    max_jump = np.concatenate(max_jumps)
    avg_jump = np.concatenate(
        avg_jumps
    )  # not necessarily linear in trajectory length because of nan padding

    mort = (
        pl.scan_parquet(
            data_dir.joinpath(f"{data_version}-tokenized", "test").joinpath(
                "outcomes.parquet"
            )
        )
        .select("same_admission_death")
        .collect()
        .to_numpy()
        .ravel()
    )

    llos = (
        pl.scan_parquet(
            data_dir.joinpath(f"{data_version}-tokenized", "test").joinpath(
                "outcomes.parquet"
            )
        )
        .select("long_length_of_stay")
        .collect()
        .to_numpy()
        .ravel()
    )

    df = pd.DataFrame.from_dict(
        {
            "traj_len": traj_len,
            "max_jump": max_jump,
            "avg_jump": avg_jump,
            "mort": mort.astype(int),
            "llos": llos.astype(int),
        }
    )

    lr_mort = smf.logit("mort ~ 1 + traj_len + max_jump + avg_jump", data=df).fit()
    logger.info(lr_mort.summary())

    lr_llos = smf.logit("llos ~ 1 + traj_len + max_jump + avg_jump", data=df).fit()
    logger.info(lr_llos.summary())


if __name__ == "__main__":
    fi.Fire(main)
