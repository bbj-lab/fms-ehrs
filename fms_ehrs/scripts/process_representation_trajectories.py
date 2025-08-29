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

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

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
    run_stats: bool = False,
):
    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (data_dir, model_loc)
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
        ).astype(np.float16)  # np.load(f) will have shape n_obs × tl_len × d_rep

        jumps = np.concatenate(
            Parallel(n_jobs=-1, verbose=True)(
                delayed(get_jumps_from_shard)(f)
                for f in tqdm.tqdm(featfiles, desc="shards")
            )
        )  # shape n_obs × tl_len-1

    if save_jumps:
        set_perms(np.save)(
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
    if run_stats:
        traj_len = np.nansum(jumps.astype(np.float64), axis=-1)  # prevent overflow
        max_jump = np.nanmax(jumps, axis=-1)
        avg_jump = np.nanmean(
            jumps.astype(np.float64), axis=-1
        )  # not necessarily linear in trajectory length because of nan padding

        outcomes = (
            "same_admission_death",
            "long_length_of_stay",
            "icu_admission",
            "icu_admission_24h",
            "imv_event",
            "imv_event_24h",
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

        res["same_admission_death_24h"] = np.zeros_like(res["same_admission_death"])
        res["long_length_of_stay_24h"] = np.zeros_like(res["same_admission_death"])

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
            if not outcome.endswith("_24h"):
                logger.info(outcome)
                lr[outcome] = smf.logit(
                    f"{outcome} ~ 1 + traj_len + max_jump + anom_scr",
                    data=df.loc[lambda x: x[outcome + "_24h"] == 0],
                ).fit()
                logger.info(lr[outcome].summary())
                logger.info(lr[outcome].summary().as_latex())


if __name__ == "__main__":
    fi.Fire(main)
