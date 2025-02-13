#!/usr/bin/env python3

"""
learn the tokenizer on the training set and apply it to the validation and test
sets
"""

import pathlib

from tokenizer import ClifTokenizer, summarize

for cut_at_24h in (False, True):

    data_version = "day_stays_qc" + ("_first_24h" if cut_at_24h else "")
    max_seq_length = 1024
    day_stay_filter = True

    verbose = True
    splits = ("train", "val", "test")

    hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()
    data_dirs = dict()
    out_dirs = dict()
    for s in splits:
        data_dirs[s] = hm.joinpath("clif-data", "raw", s)
        out_dirs[s] = hm.joinpath("clif-data", f"{data_version}-tokenized", s)
        out_dirs[s].mkdir(exist_ok=True, parents=True)

    # tokenize training set
    tkzr = ClifTokenizer(
        data_dir=data_dirs["train"],
        vocab_path=(
            hm.joinpath("clif-data", "day_stays_qc-tokenized", "train", "vocab.gzip")
            if cut_at_24h
            else None
        ),
        max_seq_length=max_seq_length,
        day_stay_filter=day_stay_filter,
        cut_at_24h=cut_at_24h,
    )
    tokens_timelines = tkzr.get_tokens_timelines()

    if verbose:
        print("train".upper().ljust(81, "-"))
        tkzr.print_aux()
        summarize(tkzr, tokens_timelines)

    tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
    tokens_timelines.write_parquet(
        out_dirs["train"].joinpath("tokens_timelines.parquet")
    )
    tkzr.vocab.save(out_dirs["train"].joinpath("vocab.gzip"))

    # take the learned tokenizer and tokenize the validation and test sets
    for s in ("val", "test"):
        tkzr = ClifTokenizer(
            data_dir=data_dirs[s],
            vocab_path=(
                hm.joinpath(
                    "clif-data", "day_stays_qc-tokenized", "train", "vocab.gzip"
                )
            ),
            max_seq_length=max_seq_length,
            day_stay_filter=day_stay_filter,
            cut_at_24h=cut_at_24h,
        )
        tokens_timelines = tkzr.get_tokens_timelines()

        if verbose:
            print(s.upper().ljust(81, "-"))
            summarize(tkzr, tokens_timelines)

        tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
        tokens_timelines.write_parquet(out_dirs[s].joinpath("tokens_timelines.parquet"))

"""
examine results
"""

import numpy as np
import polars as pl
import vocabulary as vocab

v = vocab.Vocabulary().load(
    hm.joinpath("clif-data", "day_stays_qc-tokenized", "train", "vocab.gzip")
)

df = (
    pl.scan_parquet(
        hm.joinpath(
            "clif-data",
            "day_stays_qc-tokenized",
            "train",
            "tokens_timelines.parquet",
        )
    )
    .select(
        mort=pl.col("tokens").list.contains(v("expired")),
        mort_trunc=pl.col("padded").list.contains(v("expired")),
        orig_len=pl.col("tokens").list.len(),
    )
    .collect()
)

mort = df.select("mort").to_numpy().ravel()
mort_trunc = df.select("mort_trunc").to_numpy().ravel()

# anytime there's a death in the truncated set, there should be one in the original
assert np.where(mort_trunc, mort, True).all()

print("Mortality: {}".format(mort.sum()))
print("Mortality seen by model: {}".format(mort_trunc.sum()))


df.select("orig_len").describe()
df.filter(pl.col("mort")).select("orig_len").describe()
