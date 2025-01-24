#!/usr/bin/env python3

"""
learn the tokenizer on the training set and apply it to the validation and test
sets
"""

import pathlib

from tokenizer import ClifTokenizer

splits = ("train", "val", "test")

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()
data_dirs = dict()
data_dirs["train"] = hm.joinpath("clif-training-set")
data_dirs["val"] = hm.joinpath("clif-validation-set")
data_dirs["test"] = hm.joinpath("clif-test-set")

out_dirs = dict()
for s in splits:
    out_dirs[s] = data_dirs[s].parent.joinpath(data_dirs[s].stem + "-tokenized")
    out_dirs[s].mkdir(exist_ok=True)

# tokenize training set
tkzr = ClifTokenizer(data_dir=data_dirs["train"])
tokens_timelines = tkzr.get_tokens_timelines()
tokens_timelines.write_parquet(out_dirs["train"].joinpath("tokens_timelines.parquet"))
tkzr.vocab.save(out_dirs["train"].joinpath("vocab.gzip"))

# take the learned tokenizer and tokenize the validation and test sets
for s in ("val", "test"):
    tkzr = ClifTokenizer(
        data_dir=data_dirs[s], vocab_path=out_dirs["train"].joinpath("vocab.gzip")
    )
    tokens_timelines = tkzr.get_tokens_timelines()
    tokens_timelines.write_parquet(out_dirs[s].joinpath("tokens_timelines.parquet"))
