#!/usr/bin/env python3

"""
for a list of models, collect predictions and compare performance
"""

import argparse
import pathlib
import pickle

import numpy as np
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.tokenizer import Tokenizer21

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="Y21_first_24h")
parser.add_argument("--proto_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument(
    "--model_loc", type=pathlib.Path, default="../../mdls-archive/gemma-5635921-Y21"
)
parser.add_argument(
    "--outcomes",
    nargs="+",
    default=[
        "same_admission_death",
        "long_length_of_stay",
        # "ama_discharge",
        # "hospice_discharge",
    ],
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir, proto_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.out_dir, args.proto_dir, args.model_loc),
)

# load and prep data
splits = ("train", "val", "test")
data_dirs = {s: data_dir / f"{args.data_version}-tokenized" / s for s in splits}


tt = pl.scan_parquet(data_dirs["train"] / "tokens_timelines.parquet")

tkzr = Tokenizer21(
    data_dir=data_dirs["train"],
    vocab_path=data_dirs["train"] / "vocab.gzip",
    config_file=data_dirs["train"] / "config.yaml",
)

# with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=100):
#     print(
#         tt.select(
#             pl.col("tokens")
#             .explode()
#             .replace_strict(tkzr.vocab.reverse, return_dtype=pl.String)
#         )
#         .with_columns(
#             pairs=pl.concat_str(
#                 [pl.col("tokens"), pl.col("tokens").shift(-1)], separator=","
#             )
#         )
#         .filter(pl.col("tokens") != "TL_END")
#         .group_by("pairs")
#         .len()
#         .sort("len", descending=True)
#         .head(100)
#         .collect()
#     )

# with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=100):
#     print(
#         tt.select(
#             pl.col("tokens")
#             .explode()
#             .replace_strict(tkzr.vocab.reverse, return_dtype=pl.String)
#         )
#         .with_columns(
#             triplets=pl.concat_str(
#                 [
#                     pl.col("tokens"),
#                     pl.col("tokens").shift(-1),
#                     pl.col("tokens").shift(-2),
#                 ],
#                 separator=",",
#             )
#         )
#         .filter(pl.col("tokens") != "TL_END")
#         .filter(pl.col("tokens").shift(-1) != "TL_END")
#         .group_by("triplets")
#         .len()
#         .sort("len", descending=True)
#         .head(100)
#         .collect()
#     )


# tt = tt.with_columns(
#     events=pl.struct(["times", "tokens"]).map_elements(
#         lambda s: [
#             [tok for tok, _ in group]
#             for _, group in itertools.groupby(
#                 zip(s["tokens"], s["times"]), key=lambda x: x[1]
#             )
#         ]
#     )
# )

with open(
    proto_dir
    / (args.data_version + "-tokenized")
    / "train"
    / ("lda-gmm-protos-" + model_loc.stem + ".pkl"),
    "rb",
) as fp:
    pkl = pickle.load(fp)
    scaler = pkl["scaler"]
    models = pkl["models"]

reps = np.load(data_dirs["test"] / "features-{m}.npy".format(m=model_loc.stem))
tto = pl.read_parquet(
    data_dirs["test"] / "tokens_timelines_outcomes.parquet"
).with_columns(
    **{
        f"wt_md_{outcome}": models[outcome].predict_proba(scaler.transform(reps))[
            :, np.argmax(models[outcome].weights_)
        ]
        for outcome in args.outcomes
    }
)

for outcome in args.outcomes:
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, fmt_str_lengths=100):
        print(outcome)
        print(
            top10 := tto.filter(outcome)
            .filter("icu_admission_24h")
            .sort("wt_md_" + outcome, descending=True)
            .head(10)
        )
        print(top10.select("hospitalization_id").to_series().to_list())


mask = tto.select("icu_admission_24h").to_series().to_numpy()
