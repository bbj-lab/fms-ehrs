#!/usr/bin/env python3

"""
learn the tokenizer on the training set and apply it to the validation and test
sets
"""

import pathlib

from tokenizer import ClifTokenizer, summarize

verbose = True
data_version = "first-24h"
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
    vocab_path=hm.joinpath("clif-data", "day-stays-tokenized", "train", "vocab.gzip"),
    max_seq_length=1024,
    day_stay_filter=True,
    cut_at_24h=True,
)
tokens_timelines = tkzr.get_tokens_timelines()

if verbose:
    print("train".upper().ljust(81, "-"))
    tkzr.print_aux()
    summarize(tkzr, tokens_timelines)

tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
tokens_timelines.write_parquet(out_dirs["train"].joinpath("tokens_timelines.parquet"))
tkzr.vocab.save(out_dirs["train"].joinpath("vocab.gzip"))

# take the learned tokenizer and tokenize the validation and test sets
for s in ("val", "test"):
    tkzr = ClifTokenizer(
        data_dir=data_dirs[s],
        vocab_path=hm.joinpath(
            "clif-data", "day-stays-tokenized", "train", "vocab.gzip"
        ),
        max_seq_length=1024,
        day_stay_filter=True,
        cut_at_24h=True,
    )
    tokens_timelines = tkzr.get_tokens_timelines()

    if verbose:
        print(s.upper().ljust(81, "-"))
        summarize(tkzr, tokens_timelines)

    tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
    tokens_timelines.write_parquet(out_dirs[s].joinpath("tokens_timelines.parquet"))
