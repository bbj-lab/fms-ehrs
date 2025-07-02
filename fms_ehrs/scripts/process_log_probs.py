#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import collections
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl

from fms_ehrs.framework.logger import get_logger, log_summary
from fms_ehrs.framework.plotting import imshow_text, plot_histograms
from fms_ehrs.framework.util import extract_examples, collate_events_info
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir_orig", type=pathlib.Path, default="../../clif-data")
parser.add_argument("--name_orig", type=str, default="MIMIC")
parser.add_argument("--data_dir_new", type=pathlib.Path, default="../../clif-data-ucmc")
parser.add_argument("--name_new", type=str, default="UCMC")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../clif-mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument(
    "--samp_orig",
    type=str,
    nargs="*",
    default=["20826893", "27726633", "26624012", "24410460", "29173149"],
)
parser.add_argument(
    "--samp_new",
    type=str,
    nargs="*",
    default=["27055120", "792481", "12680680", "9468768", "8797520"],
)
parser.add_argument(
    "--aggregation", choices=["sum", "max", "perplexity"], default="sum"
)
parser.add_argument("--n_egs", type=int, default=10)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir_orig, data_dir_new, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_orig, args.data_dir_new, args.model_loc, args.out_dir),
)

rng = np.random.default_rng(42)
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

outl = {
    v: np.load(
        data_dirs[v]["test"].joinpath(
            "features-outliers-{m}.npy".format(m=model_loc.stem)
        )
    )
    for v in versions
}

anom = {
    v: np.load(
        data_dirs[v]["test"].joinpath(
            "features-anomaly-score-{m}.npy".format(m=model_loc.stem)
        )
    )
    for v in versions
}

vocab = Vocabulary().load(data_dirs["orig"]["train"].joinpath("vocab.gzip"))

infm = {
    v: np.load(
        data_dirs[v]["test"].joinpath("log_probs-{m}.npy".format(m=model_loc.stem)),
    )
    / -np.log(2)
    for v in versions
}

ent = {v: np.nanmean(infm[v], axis=1) for v in versions}
inf_sum = {v: np.nansum(infm[v], axis=1) for v in versions}

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

tm = {
    v: pl.scan_parquet(
        data_dirs[v]["test"].joinpath(
            "tokens_timelines_outcomes.parquet",
        )
    )
    .select("times")
    .collect()
    .to_series()
    .to_list()
    for v in versions
}

samp = {"orig": args.samp_orig, "new": args.samp_new}

ids = {
    v: np.array(
        pl.scan_parquet(
            data_dirs[v]["test"].joinpath(
                "tokens_timelines_outcomes.parquet",
            )
        )
        .select("hospitalization_id")
        .collect()
        .to_series()
        .to_numpy()
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
    extract_examples(
        timelines=tl[v],
        criteria=infm[v],
        flags=flags[v],
        vocab=vocab,
        logger=logger,
        k=args.n_egs,
        ids=ids[v],
    )

n_cols = 6
max_len = 1002
for v in versions:
    for s in samp[v]:
        i = np.argmax(s == ids[v])
        tms_i = tm[v][i]
        event_info, idx = collate_events_info(
            np.array(tms_i), np.nan_to_num(infm[v][i][: len(tms_i)]), args.aggregation
        )
        ev_inf_i = np.concatenate(
            [event_info[idx], np.zeros(len(tl[v][i]) - len(tms_i))]
        )
        tt = np.array(
            [
                (
                    (d if len(d) <= 23 else f"{d[:13]}..{d[-7:]}")
                    if (d := vocab.reverse[t]) is not None
                    else "None"
                )
                for t in tl[v][i]
            ]
        )
        imshow_text(
            values=np.nan_to_num(infm[v][i])[:max_len].reshape((-1, n_cols)),
            text=tt[:max_len].reshape((-1, n_cols)),
            title=f"Information by token for patient {s} in {names[v]}",
            savepath=out_dir.joinpath(
                "tokens-{v}-{s}-{m}-hist.pdf".format(v=v, s=s, m=model_loc.stem)
            ),
        )
        imshow_text(
            values=ev_inf_i[:max_len].reshape((-1, n_cols)),
            text=tt[:max_len].reshape((-1, n_cols)),
            title=f"Information by event for patient {s} in {names[v]}",
            savepath=out_dir.joinpath(
                "events-{agg}-{v}-{s}-{m}-hist.pdf".format(
                    agg=args.aggregation, v=v, s=s, m=model_loc.stem
                )
            ),
        )


for v in versions:
    fig = px.scatter(
        pd.DataFrame({"anomaly score": anom[v], "information": inf_sum[v]}),
        x="information",
        y="anomaly score",
        trendline="ols",
        color_discrete_sequence=["#DE7C00"],
    )
    fig.data[1].line.color = "#789D4A"
    fig.update_layout(
        title="Anomaly score vs. sum of information", template="plotly_white"
    )
    fig.update_traces(marker=dict(size=3))
    fig.write_image(
        out_dir.joinpath("anom-ent-{m}-{v}-hist.pdf".format(m=model_loc.stem, v=v))
    )


logger.info("---fin")
