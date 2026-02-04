#!/usr/bin/env python3

"""
Compute norms of jumps
"""

import argparse
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
parser.add_argument("--data_dir", type=pathlib.Path, default=None)
parser.add_argument("--data_version", type=str, default="day_stays_first_24h")
parser.add_argument("--model_loc", type=pathlib.Path, default=None)
parser.add_argument("--save_jumps", action="store_true")
parser.add_argument("--load_jumps", action="store_true")
parser.add_argument("--run_stats", action="store_true")
parser.add_argument("--all_layers", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

featfiles = sorted(
    (data_dir / f"{args.data_version}-tokenized" / "test").glob(
        "all-features{x}-{m}-batch*.npy".format(
            x="-all-layers" if args.all_layers else "", m=model_loc.stem
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
        delayed(get_jumps_from_shard)(f) for f in tqdm.tqdm(featfiles, desc="shards")
    )
)  # shape n_obs × tl_len-1 if not `all_layers` else  n_obs × tl_len -1 × n_layers + 1

if args.save_jumps:
    set_perms(np.save)(
        data_dir
        / f"{args.data_version}-tokenized"
        / "test"
        / "all-jumps{x}-{m}.npy".format(
            x="-all-layers" if args.all_layers else "", m=model_loc.stem
        ),
        jumps,
    )

logger.info("---fin")
