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
from joblib import Parallel, delayed

from logger import get_logger

# from vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    data_dir: os.PathLike = None,
    data_version: str = "day_stays_first_24h",
    model_loc: os.PathLike = None,
    save_jumps: bool = False,
    load_jumps: bool = False,
):

    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (data_dir, model_loc),
    )

    if load_jumps:
        jumps = np.load(
            data_dir.joinpath(
                f"{data_version}-tokenized",
                "test",
                "all-jumps-{m}.npy".format(m=model_loc.stem),
            )
        )

    else:
        featfiles = sorted(
            data_dir.joinpath(f"{data_version}-tokenized", "test").glob(
                "all-features-{m}-batch*.npy".format(m=model_loc.stem)
            ),
            key=lambda s: int(s.stem.split("batch")[-1]),
        )

        get_jumps_from_shard = lambda f: np.linalg.norm(
            np.diff(np.load(f), axis=1), axis=-1
        ).astype(
            np.float16
        )  # np.load(f) will have shape n_obs × tl_len × d_rep

        jumps = np.concatenate(
            Parallel(n_jobs=-1, verbose=True)(
                delayed(get_jumps_from_shard)(f)
                for f in tqdm.tqdm(featfiles, desc="shards")
            )
        )  # shape n_obs × tl_len-1

    if save_jumps:
        np.save(
            data_dir.joinpath(
                f"{data_version}-tokenized",
                "test",
                "all-jumps-{m}.npy".format(m=model_loc.stem),
            ),
            jumps,
        )

    """
    are trajectory statistics predictive of outcomes?
    """

    traj_len = np.nansum(jumps.astype(np.float64), axis=-1)  # prevent overflow
    max_jump = np.nanmax(jumps, axis=-1)
    avg_jump = np.nanmean(
        jumps.astype(np.float64), axis=-1
    )  # not necessarily linear in trajectory length because of nan padding

    outcomes = (
        "same_admission_death",
        "long_length_of_stay",
        "icu_admission",
        "imv_event",
    )
    res = dict()
    for outcome in outcomes:
        res[outcome] = (
            pl.scan_parquet(
                data_dir.joinpath(
                    f"{data_version}-tokenized",
                    "test",
                    "tokens_timelines_outcomes.parquet",
                )
            )
            .select(outcome)
            .collect()
            .to_numpy()
            .ravel()
            .astype(int)
        )

    anom = np.load(
        data_dir.joinpath(
            f"{data_version}-tokenized",
            "test",
            "features-anomaly-score-{m}.npy".format(m=model_loc.stem),
        )
    )

    df = pd.DataFrame.from_dict(
        {
            "traj_len": traj_len,
            "max_jump": max_jump,
            "avg_jump": avg_jump,
            "anom_scr": anom,
        }
        | res
    )

    lr = dict()
    for outcome in outcomes:
        logger.info(outcome)
        lr[outcome] = smf.logit(
            f"{outcome} ~ 1 + traj_len + max_jump + anom_scr", data=df
        ).fit()
        logger.info(lr[outcome].summary())

    """
    what do large jumps look like, tokenwise?
    """

    # vocab = Vocabulary().load(
    #     data_dir.joinpath(f"{data_version}-tokenized", "train", "vocab.gzip")
    # )
    #
    # k = 25
    # w_sz = 5
    # top_k_flat_idx = np.argsort(np.nan_to_num(jumps.flatten()))[::-1][:k]
    # top_k_idx = np.array(np.unravel_index(top_k_flat_idx, jumps.shape)).T
    #
    # raw_padded_timelines = np.array(
    #     pl.scan_parquet(
    #         data_dir.joinpath(
    #             f"{data_version}-tokenized", "test", "tokens_timelines.parquet"
    #         )
    #     )
    #     .select("padded")
    #     .collect()
    #     .to_series()
    #     .to_list()
    # )
    #
    # m = raw_padded_timelines.shape[-1]
    #
    # for i0, i1 in top_k_idx:
    #     ints = raw_padded_timelines[i0, max(0, i1 - w_sz) : min(m - 1, i1 + w_sz)]
    #     tkns = "->".join(vocab.reverse[i] for i in ints)
    #     hit = vocab.reverse[raw_padded_timelines[i0, i1 + 1]]
    #     logger.info(
    #         f"{i0=}, {i1=} "
    #     )
    #     logger.info(f"{hit=} in {tkns}")


if __name__ == "__main__":
    fi.Fire(main)
