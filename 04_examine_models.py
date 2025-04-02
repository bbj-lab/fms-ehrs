#!/usr/bin/env python3

"""
load a model and make some plots
"""

import functools
import os
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

from logger import get_logger
from vocabulary import Vocabulary

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
    *,
    projector_type: typing.Literal["PCA", "TSNE"] = "PCA",
    data_dir: os.PathLike = None,
    data_version: str = None,
    ref_mdl_loc: os.PathLike = None,
    addl_mdls_loc: str = None,
    out_dir: os.PathLike = None,
):

    data_dir, ref_mdl_loc, out_dir = map(
        lambda d: pathlib.Path(d).expanduser().resolve(),
        (data_dir, ref_mdl_loc, out_dir),
    )

    addl_mdls_loc = (
        [
            pathlib.Path(d.strip()).expanduser().resolve()
            for d in addl_mdls_loc.split(",")
        ]
        if addl_mdls_loc is not None
        else None
    )

    train_dir = data_dir.joinpath(f"{data_version}-tokenized", "train")

    vocab = Vocabulary().load(train_dir.joinpath("vocab.gzip"))
    ref_mdl = AutoModelForCausalLM.from_pretrained(ref_mdl_loc)

    addl_mdls = [AutoModelForCausalLM.from_pretrained(d) for d in addl_mdls_loc]

    """
    dimensionality reduction on token embeddings
    """

    # size: vocab x emb_dim
    emb = ref_mdl.get_input_embeddings()(t_arange(len(vocab)))
    projector = (
        PCA(n_components=2, random_state=42)
        if projector_type == "PCA"
        else TSNE(n_components=2, random_state=42, perplexity=150)
    )
    proj = projector.fit_transform(emb.detach())
    if projector_type == "PCA":
        logger.info(f"{projector.explained_variance_ratio_=}")

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

    for i, m in enumerate(addl_mdls):
        addl_emb = m.get_input_embeddings()(t_arange(len(vocab)))
        addl_proj = projector.transform(addl_emb.detach())
        addl_df = (
            pl.from_numpy(data=addl_proj, schema=["dim1", "dim2"])
            .with_columns(token=pl.Series(vocab.lookup.keys()))
            .with_columns(
                type=pl.col("token").map_elements(
                    key_type,
                    return_dtype=pl.String,
                    skip_nulls=False,
                )
            )
        )

        addl_fig = go.Scatter(
            x=addl_df["dim1"],
            y=addl_df["dim2"],
            mode="markers",
            marker=dict(size=4, color=("black", "grey")[i], symbol=("x", "cross")[i]),
            text=addl_df["type"],
            hoverinfo="text",
            name=addl_mdls_loc[i].stem.split("-")[-1],
        )

        fig.add_trace(addl_fig)

    fig.write_html(
        out_dir.joinpath("embedding_vis-{m}.html".format(m=ref_mdl_loc.stem))
    )

    """
    quantile embeddings only
    """

    # size: vocab x emb_dim
    emb = ref_mdl.get_input_embeddings()(t_arange(10))
    projector = (
        PCA(n_components=2, random_state=42)
        if projector_type == "PCA"
        else TSNE(n_components=2, random_state=42, perplexity=150)
    )
    proj = projector.fit_transform(emb.detach())
    if projector_type == "PCA":
        logger.info(f"{projector.explained_variance_ratio_=}")

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

    for i, m in enumerate(addl_mdls):
        addl_emb = m.get_input_embeddings()(t_arange(10))
        addl_proj = projector.transform(addl_emb.detach())
        addl_df = (
            pl.from_numpy(data=addl_proj, schema=["dim1", "dim2"])
            .with_columns(token=pl.Series(list(vocab.lookup.keys())[:10]))
            .with_columns(
                type=pl.col("token").map_elements(
                    key_type,
                    return_dtype=pl.String,
                    skip_nulls=False,
                )
            )
        )

        addl_fig = go.Scatter(
            x=addl_df["dim1"],
            y=addl_df["dim2"],
            mode="markers",
            marker=dict(size=4, color=("black", "grey")[i], symbol=("x", "cross")[i]),
            text=addl_df["type"],
            hoverinfo="text",
            name=addl_mdls_loc[i].stem.split("-")[-1],
        )

        fig.add_trace(addl_fig)

    fig.write_html(out_dir.joinpath("embedding_q-{m}.html".format(m=ref_mdl_loc.stem)))


if __name__ == "__main__":
    fi.Fire(main)
