#!/usr/bin/env python3

"""
load a model and make some plots
"""

import functools
import pathlib
import re
import typing

import fire as fi
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import arange as t_arange
from transformers import AutoModelForCausalLM

from vocabulary import Vocabulary
from logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@functools.cache
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


@logger.log_calls
def main(
    projector_type: typing.Literal["PCA", "TSNE"] = "PCA",
    train_dir=pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").joinpath(
        "clif-data", f"day_stays_qc_first_24h-tokenized", "train"
    ),
    model_loc=pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").joinpath(
        "clif-mdls-archive",
        "llama-57218445-run3",
        # "medium-packing-tuning-57164794-run2-ckpt-7000"
    ),
    out_dir=pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/"),
):

    train_dir, model_loc, out_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (train_dir, model_loc, out_dir),
    )

    vocab = Vocabulary().load(train_dir.joinpath("vocab.gzip"))
    model = AutoModelForCausalLM.from_pretrained(model_loc)

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
    if projector_type == "PCA":
        logger.log(f"{proj.explained_variance_ratio_=}")

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

    fig.write_html(out_dir.joinpath("embedding_vis-{m}.html".format(m=model_loc.stem)))

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
    if projector_type == "PCA":
        logger.log(f"{proj.explained_variance_ratio_=}")

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

    fig.write_html(out_dir.joinpath("embedding_q-{m}.html".format(m=model_loc.stem)))


if __name__ == "__main__":
    fi.Fire(main)
