#!/usr/bin/env python3

"""
Compute norms of jumps
"""

import os
import pathlib

import fire as fi
import numpy as np
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
    all_layers: bool = False,
):
    data_dir, model_loc = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (data_dir, model_loc)
    )

    featfiles = sorted(
        data_dir.joinpath(f"{data_version}-tokenized", "test").glob(
            "all-features{x}-{m}-batch*.npy".format(
                x="-all-layers" if all_layers else "", m=model_loc.stem
            )
        ),
        key=lambda s: int(s.stem.split("batch")[-1]),
    )

    get_jumps_from_shard = lambda f: np.linalg.norm(
        np.diff(np.load(f), axis=1), axis=1
    ).astype(
        np.float16
    )  # np.load(f) will have shape n_obs × tl_len × d_rep if not `all_layers`
    # else n_obs × tl_len × d_rep × n_layers + 1

    jumps = np.concatenate(
        Parallel(n_jobs=-1, verbose=True)(
            delayed(get_jumps_from_shard)(f)
            for f in tqdm.tqdm(featfiles, desc="shards")
        )
    )  # shape n_obs × tl_len-1 if not `all_layers` else  n_obs × tl_len -1 × n_layers + 1

    if save_jumps:
        set_perms(np.save)(
            data_dir.joinpath(
                f"{data_version}-tokenized",
                "test",
                "all-jumps{x}-{m}.npy".format(
                    x="-all-layers" if all_layers else "", m=model_loc.stem
                ),
            ),
            jumps,
        )


if __name__ == "__main__":
    fi.Fire(main)
