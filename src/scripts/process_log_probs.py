#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import collections
import logging
import pathlib

import numpy as np
import polars as pl

from src.framework.logger import get_logger
from src.framework.util import log_summary, plot_histograms
from src.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--name_orig", type=str, default="MIMIC")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../clif-data-ucmc")
parser.add_argument("--name_new", type=str, default="UCMC")
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

names = {"orig": args.name_orig, "new": args.name_new}
splits = ("train", "val", "test")
versions = ("orig", "new")
outcomes = ("same_admission_death", "long_length_of_stay", "icu_admission", "imv_event")

data_dirs = collections.defaultdict(dict)
data_dirs["orig"] = {
    s: data_dir_orig.joinpath(f"{args.data_version}-tokenized", s) for s in splits
}
data_dirs["new"] = {
    s: data_dir_new.joinpath(f"{args.data_version}-tokenized", s) for s in splits
}

vocab = Vocabulary().load(data_dirs["orig"]["train"].joinpath("vocab.gzip"))

infm = {
    v: np.load(
        data_dirs[v]["test"].joinpath("log_probs-{m}.npy".format(m=model_loc.stem)),
    )
    / -np.log(2)
    for v in versions
}

tl = {
    v: np.array(
        pl.scan_parquet(
            data_dirs[v]["test"].joinpath(
                "tokens_timelines_outcomes.parquet",
            )
        )
        .select("padded")
        .collect()
        .to_series()
        .to_list()
    )
    for v in versions
}

flags = {
    v: (
        pl.scan_parquet(
            data_dirs[v]["test"].joinpath(
                "tokens_timelines_outcomes.parquet",
            )
        )
        .with_columns(
            [
                pl.when(pl.col(outcome))
                .then(pl.lit(outcome))
                .otherwise(None)
                .alias(outcome)
                for outcome in outcomes
            ]
        )
        .with_columns(flags=pl.concat_str(outcomes, separator=", ", ignore_nulls=True))
        .select("flags")
        .collect()
        .to_series()
        .to_list()
    )
    for v in versions
}


def extract_examples(
    timelines: np.array,
    criteria: np.array,
    flags: list = None,
    vocab: Vocabulary = vocab,
    k: int = 10,
    w_sz: int = 3,
    lag: int = 0,
    logger: logging.Logger = logger,
    top_k: bool = True,
):
    assert timelines.shape[0] == criteria.shape[0]
    assert timelines.shape[1] == criteria.shape[1] + lag
    if flags:
        assert len(flags) == timelines.shape[0]
    top_k_flat_idx = (
        np.argsort(np.nan_to_num(criteria.flatten()))[::-1][:k]
        if top_k
        else np.argsort(np.nan_to_num(criteria.flatten(), nan=np.inf))[:k]  # bottom k
    )
    top_k_idx = np.array(np.unravel_index(top_k_flat_idx, criteria.shape)).T
    m = timelines.shape[-1]
    for i0, i1 in top_k_idx:
        ints = timelines[i0, max(0, i1 - w_sz) : min(m - 1, i1 + w_sz + lag)]
        tkns = "->".join(
            s if (s := vocab.reverse[i]) is not None else "None" for i in ints
        )
        hit = " ".join(
            s if (s := vocab.reverse[i]) is not None else "None"
            for i in timelines[i0][i1 : i1 + lag + 1]
        )
        if flags:
            logger.info(f"{i0=}, {i1=} | {flags[i0]}")
        else:
            logger.info(f"{i0=}, {i1=} ")
        logger.info(f"{hit=} in {tkns}")
        logger.info(
            "->".join(
                map(
                    str,
                    criteria[i0, max(0, i1 - w_sz) : min(m - 1, i1 + w_sz + lag)].round(
                        2
                    ),
                )
            )
        )


# single-token events
logger.info("Singletons |".ljust(79, "="))
plot_histograms(
    named_arrs={names[v]: infm[v] for v in versions},
    title="Histogram of tokenwise information",
    xaxis_title="bits",
    yaxis_title="frequency",
    savepath=out_dir.joinpath("log_probs-{m}-hist.pdf".format(m=model_loc.stem)),
)
for v in versions:
    logger.info(f"{names[v]}:")
    log_summary(infm[v], logger)
    extract_examples(timelines=tl[v], criteria=infm[v], flags=flags[v], logger=logger)
    logger.info("bottom k")
    extract_examples(
        timelines=tl[v],
        criteria=infm[v],
        flags=flags[v],
        logger=logger,
        k=100,
        top_k=False,
    )


# 2-token events
logger.info("Pairs |".ljust(79, "="))
infm_pairs = {
    v: np.lib.stride_tricks.sliding_window_view(infm[v], window_shape=2, axis=-1).mean(
        axis=-1
    )
    for v in versions
}

plot_histograms(
    named_arrs={names[v]: infm_pairs[v] for v in versions},
    title="Histogram of pairwise information (per token)",
    xaxis_title="bits",
    yaxis_title="frequency",
    savepath=out_dir.joinpath("log_probs_pairs-{m}-hist.pdf".format(m=model_loc.stem)),
)
for v in versions:
    logger.info(f"{names[v]}:")
    log_summary(infm_pairs[v], logger)
    extract_examples(
        timelines=tl[v], criteria=infm_pairs[v], flags=flags[v], lag=1, logger=logger
    )

# 3-token events
logger.info("Triples |".ljust(79, "="))
infm_trips = {
    v: np.lib.stride_tricks.sliding_window_view(infm[v], window_shape=3, axis=-1).mean(
        axis=-1
    )
    for v in versions
}

plot_histograms(
    named_arrs={names[v]: infm_trips[v] for v in versions},
    title="Histogram of triple information (per token)",
    xaxis_title="bits",
    yaxis_title="frequency",
    savepath=out_dir.joinpath("log_probs_trips-{m}-hist.pdf".format(m=model_loc.stem)),
)
for v in versions:
    logger.info(f"{names[v]}:")
    log_summary(infm_trips[v], logger)
    extract_examples(
        timelines=tl[v], criteria=infm_trips[v], flags=flags[v], lag=2, logger=logger
    )

# 4-token events
# logger.info("Quads |".ljust(79, "="))
# infm_quads = {
#     v: np.lib.stride_tricks.sliding_window_view(infm[v], window_shape=4, axis=-1).mean(
#         axis=-1
#     )
#     for v in versions
# }
#
# plot_histograms(
#     named_arrs={names[v]: infm_quads[v] for v in versions},
#     title="Histogram of pairwise information (per token)",
#     xaxis_title="bits",
#     yaxis_title="frequency",
#     savepath=out_dir.joinpath("log_probs_quads-{m}-hist.pdf".format(m=model_loc.stem)),
# )
# for v in versions:
#     logger.info(f"{names[v]}:")
#     log_summary(infm_quads[v], logger)
#     extract_examples(
#         timelines=tl[v], criteria=infm_quads[v], flags=flags[v], lag=3, logger=logger
#     )

logger.info("---fin")
