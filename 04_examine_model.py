#!/usr/bin/env python3

"""
load a Mamba and play with it
"""

import pathlib
import re

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import arange as t_arange
from transformers import AutoModelForCausalLM

from vocabulary import Vocabulary

projector_type = "PCA"
data_version = "day_stays_qc_first_24h"

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

train_dir = hm.joinpath("clif-data", f"{data_version}-tokenized", "train")
vocab = Vocabulary().load(train_dir.joinpath("vocab.gzip"))
model_version = "small-packed"  # "small"
model = AutoModelForCausalLM.from_pretrained(
    hm.joinpath(
        "clif-mdls",
        model_version,
        "mdl-day_stays_qc-small-packed-2025-02-18T19:25:32-06:00",
    )
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


"""
dimensionality reduction on token embeddings
"""

# size: vocab x emb_dim
emb = model.get_input_embeddings()(t_arange(len(vocab)))
projector = (
    PCA(n_components=2, random_state=42)
    if projector_type == "PCA"
    else TSNE(n_components=2, random_state=42, perplexity=150)
)
proj = projector.fit_transform(emb.detach())

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

fig.write_html(hm.joinpath("embedding_vis-{m}.html".format(m=model_version)))

"""
quantile embeddings only
"""

# size: vocab x emb_dim
emb = model.get_input_embeddings()(t_arange(10))
projector = (
    PCA(n_components=2, random_state=42)
    if projector_type == "PCA"
    else TSNE(n_components=2, random_state=42, perplexity=150)
)
proj = projector.fit_transform(emb.detach())

fig = px.scatter(
    pl.from_numpy(data=proj, schema=["dim1", "dim2"]).with_columns(
        token=pl.Series(list(vocab.lookup.keys())[:10])
    ),
    x="dim1",
    y="dim2",
    color="token",
    width=650,
    title="Quantile embedding",
    hover_name="token",
)

# connect Q0->Q1->Q2->...->Q9
fig.add_trace(
    go.Scatter(
        x=proj[:, 0],
        y=proj[:, 1],
        mode="lines",
        line=dict(color="grey", width=0.75),
        showlegend=False,
    )
)

fig.write_html(hm.joinpath("embedding_q-{m}.html".format(m=model_version)))


"""
return basic parameters from searches
"""

keys = ("hidden_size", "n_layer", "num_hidden_layers", "state_size", "vocab_size")
for s in ("small-lr-search", "smaller-lr-search", "smallest-lr-search"):
    print(s.ljust(79, "="))
    m = next(iter(hm.joinpath("clif-mdls", s).glob("**/checkpoint-*")))
    model = AutoModelForCausalLM.from_pretrained(m)
    conf = model.config.to_dict()
    for k in keys:
        print("{}: {}".format(k, conf[k]))

vocab.print_aux()
