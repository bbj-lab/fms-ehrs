#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import pathlib

import numpy as np

from src.framework.logger import get_logger
from src.framework.util import log_summary, plot_histograms
from src.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../clif-data-ucmc")
parser.add_argument("--data_version", type=str, default="QC_day_stays_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama-med-58788824",
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc, args.out_dir),
)

# load and prep data
splits = ("train", "val", "test")
data_dirs_orig = dict()
data_dirs_new = dict()
for s in splits:
    data_dirs_orig[s] = data_dir_orig.joinpath(f"{args.data_version}-tokenized", s)
    data_dirs_new[s] = data_dir_new.joinpath(f"{args.data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs_orig["train"].joinpath("vocab.gzip"))

inf_orig = np.load(
    data_dirs_orig["test"].joinpath("log_probs-{m}.npy".format(m=model_loc.stem)),
) / -np.log(2)
inf_new = np.load(
    data_dirs_new["test"].joinpath("log_probs-{m}.npy".format(m=model_loc.stem)),
) / -np.log(2)

logger.info("Singletons...")
plot_histograms(
    named_arrs={"MIMIC": inf_orig, "UCMC": inf_new},
    title="Histogram of tokenwise information",
    xaxis_title="bits",
    yaxis_title="frequency",
    savepath=out_dir.joinpath("log_probs-{m}-hist.pdf".format(m=model_loc.stem)),
)
logger.info("Orig:")
log_summary(inf_orig, logger)
logger.info("New:")
log_summary(inf_new, logger)

logger.info("Pairs...")
inf_orig_pairs = np.lib.stride_tricks.sliding_window_view(
    inf_orig, window_shape=2, axis=-1
).sum(axis=-1)
inf_new_pairs = np.lib.stride_tricks.sliding_window_view(
    inf_new, window_shape=2, axis=-1
).sum(axis=-1)
plot_histograms(
    named_arrs={"MIMIC": inf_orig_pairs, "UCMC": inf_new_pairs},
    title="Histogram of pairwise information",
    xaxis_title="bits",
    yaxis_title="frequency",
    savepath=out_dir.joinpath("log_probs_pairs-{m}-hist.pdf".format(m=model_loc.stem)),
)
logger.info("Orig:")
log_summary(inf_orig_pairs, logger)
logger.info("New:")
log_summary(inf_new_pairs, logger)

logger.info("Triples...")
inf_orig_trips = np.lib.stride_tricks.sliding_window_view(
    inf_orig, window_shape=3, axis=-1
).sum(axis=-1)
inf_new_trips = np.lib.stride_tricks.sliding_window_view(
    inf_new, window_shape=3, axis=-1
).sum(axis=-1)
plot_histograms(
    named_arrs={"MIMIC": inf_orig_trips, "UCMC": inf_new_trips},
    title="Histogram of triplet information",
    xaxis_title="bits",
    yaxis_title="frequency",
    savepath=out_dir.joinpath("log_probs_trips-{m}-hist.pdf".format(m=model_loc.stem)),
)
logger.info("Orig:")
log_summary(inf_orig_trips, logger)
logger.info("New:")
log_summary(inf_new_trips, logger)

logger.info("---fin")
