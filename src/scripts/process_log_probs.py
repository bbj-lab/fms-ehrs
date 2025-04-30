#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import pathlib

import numpy as np
from src.framework.util import log_summary, plot_histogram

from src.framework.logger import get_logger
from src.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama-med-58788824",
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../")
parser.add_argument("--batch_sz", type=int, default=2**8)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir),
)

# load and prep data
splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = data_dir.joinpath(f"{args.data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

log_probs = np.load(
    data_dirs[s].joinpath("log_probs-{m}.npy".format(m=model_loc.stem)),
)  # n_obs × tl_len

log_probs /= -np.log(
    2
)  # torch & numpy give us natural logs, so we do a change-of-basis

logger.info("Singletons...")
plot_histogram(
    log_probs,
    "Histogram of tokenwise information",
    savepath=out_dir.joinpath(
        "log_probs-{m}-hist-{d}.pdf".format(m=model_loc.stem, d=data_dir.stem)
    ),
)
log_summary(log_probs, logger)

logger.info("Pairs...")
logp_pairs = np.lib.stride_tricks.sliding_window_view(
    log_probs, window_shape=2, axis=-1
).sum(axis=-1)
plot_histogram(
    logp_pairs,
    "Histogram of pairwise information",
    savepath=out_dir.joinpath(
        "log_probs_pairs-{m}-hist-{d}.pdf".format(m=model_loc.stem, d=data_dir.stem)
    ),
)
log_summary(logp_pairs, logger)

logger.info("Triples...")
logp_trips = np.lib.stride_tricks.sliding_window_view(
    log_probs, window_shape=3, axis=-1
).sum(axis=-1)
plot_histogram(
    logp_trips,
    "Histogram of triplet information",
    savepath=out_dir.joinpath(
        "log_probs_trips-{m}-hist-{d}.pdf".format(m=model_loc.stem, d=data_dir.stem)
    ),
)
log_summary(logp_trips, logger)

logger.info("---fin")
