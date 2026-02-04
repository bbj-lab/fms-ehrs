#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import json
import pathlib

import numpy as np
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import imshow_text
from fms_ehrs.framework.vocabulary import Vocabulary

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
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument(
    "--samp",
    type=str,
    nargs="*",
    default=["20826893", "27726633", "26624012", "24410460", "29173149"],
)
parser.add_argument(
    "--aggregation", choices=["sum", "max", "perplexity"], default="sum"
)
parser.add_argument("--n_egs", type=int, default=10)
parser.add_argument("--max_len", type=int, default=102)
parser.add_argument("--emit_json", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir),
)

rng = np.random.default_rng(42)
splits = ("train", "val", "test")
data_dirs = {s: data_dir / f"{args.data_version}-tokenized" / s for s in splits}
vocab = Vocabulary().load(data_dirs["train"] / "vocab.gzip")
infm = data_dirs[v]["test"] / "information-{m}.npy".format(m=model_loc.stem)

logger.info(f"{v=},{np.nanmean(infm)=}")


tl = np.array(
    pl.scan_parquet(data_dirs[v]["test"] / "tokens_timelines.parquet")
    .select("padded")
    .collect()
    .to_series()
    .to_list()
)

tm = (
    pl.scan_parquet(data_dirs[v]["test"] / "tokens_timelines.parquet")
    .select("times")
    .collect()
    .to_series()
    .to_list()
)

ids = np.array(
    pl.scan_parquet(data_dirs[v]["test"] / "tokens_timelines.parquet")
    .select("hospitalization_id")
    .collect()
    .to_series()
    .to_numpy()
)

n_cols = 6
n_rows = args.max_len // n_cols
max_len = n_rows * n_cols
height = (700 * n_rows) // 42

listing = []

for s in args.samp:
    i = np.argmax(s == ids[v])
    tms_i = tm[v][i][:max_len]
    tt = np.array(
        [
            (
                (d if len(d) <= 23 else f"{d[:13]}..{d[-7:]}")
                if (d := vocab.reverse[t]) is not None
                else "None"
            )
            for t in tl[i]
        ]
    )
    imshow_text(
        values=np.nan_to_num(infm[v][i])[:max_len].reshape((-1, n_cols)),
        text=tt[:max_len].reshape((-1, n_cols)),
        # title=f"Information by token for patient {s} in {names[v]}",
        savepath=out_dir
        / "tokens-{v}-{s}-{m}-hist.pdf".format(v=v, s=s, m=model_loc.stem),
        autosize=False,
        zmin=0,
        height=height,
        width=1000,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    listing.append(
        {
            "id": s,
            "information": list(np.nan_to_num(infm[v][i]))[:max_len],
            "tokens": list(tl[v][i])[:max_len],
        }
    )

if args.emit_json:
    json.dumps(listing)


logger.info("---fin")
