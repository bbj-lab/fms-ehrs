#!/usr/bin/env python3

"""
load a Mamba and play with it
"""


import pathlib
import re

from transformers import AutoModelForCausalLM
from datasets import load_dataset
from torch import arange as t_arange
from sklearn.manifold import TSNE

import polars as pl
import plotly.express as px

from vocabulary import Vocabulary

data_version = "day-stays"
model_version = "neftune"

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = hm.joinpath("clif-data", f"{data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
output_dir = hm.joinpath("clif-mdls", model_version)

model = AutoModelForCausalLM.from_pretrained(output_dir.joinpath("checkpoint-1500"))

dataset = load_dataset(
    "parquet",
    data_files={
        s: str(data_dirs[s].joinpath("tokens_timelines.parquet"))
        for s in ("train", "val", "test")
    },
)


def key_type(word: str) -> str:
    if word is None:
        return "OTHER"
    if re.fullmatch(r"Q\d", word):
        return "Q"
    if (pre := word.split("_")[0]) in ("LAB", "VTL", "MED", "ASMT"):
        return pre
    if word in ("TL_START", "TL_END", "PAD", "TRUNC"):
        return "SPECIAL"
    return "OTHER"


# size: vocab x emb_dim
emb = model.get_input_embeddings()(t_arange(len(vocab)))
proj = TSNE(n_components=2, random_state=42, perplexity=10).fit_transform(emb.detach())

df = (
    pl.from_numpy(data=proj, schema=["dim1", "dim2"])
    .with_columns(token=pl.Series(vocab.lookup.keys()))
    .with_columns(
        type=pl.col("token").map_elements(
            key_type,
            return_dtype=pl.String,
            skip_nulls=False,
        )
    )
)

fig = px.scatter(
    df,
    x="dim1",
    y="dim2",
    color="type",
    width=650,
    title="Token embedding",
    hover_name="token",
)

fig.write_html(hm.joinpath("embedding_vis.html"))
