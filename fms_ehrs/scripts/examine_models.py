#!/usr/bin/env python3

"""
load models and plot embeddings
"""

import os
import pathlib
import typing

import fire as fi
from pacmap import PaCMAP
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import arange as t_arange
from transformers import AutoModelForCausalLM

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.plotting import colors
from fms_ehrs.framework.tokenizer import token_type
from fms_ehrs.framework.vocabulary import Vocabulary

pio.kaleido.scope.mathjax = None

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    projector_type: typing.Literal["PCA", "TSNE", "PACMAP"] = "PCA",
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
        else []
    )

    train_dir = data_dir.joinpath(f"{data_version}-tokenized", "train")

    vocab = Vocabulary().load(train_dir.joinpath("vocab.gzip"))
    ref_mdl = AutoModelForCausalLM.from_pretrained(ref_mdl_loc)

    addl_mdls = [AutoModelForCausalLM.from_pretrained(d) for d in addl_mdls_loc]

    """
    dimensionality reduction on token embeddings
    """

    # size: vocab × emb_dim
    emb = ref_mdl.get_input_embeddings()(t_arange(len(vocab)))
    match projector_type:
        case "PCA":
            projector = PCA(n_components=2, random_state=42)
        case "TSNE":
            projector = TSNE(n_components=2, random_state=42, perplexity=150)
        case "PACMAP":
            projector = PaCMAP(n_components=2, n_neighbors=None, random_state=42)
        case _:
            raise Exception(f"{projector_type=} unsupported")
    proj = projector.fit_transform(emb.detach().numpy())
    if projector_type == "PCA":
        logger.info(f"{projector.explained_variance_ratio_=}")

    df = (
        pl.from_numpy(data=proj, schema=["dim1", "dim2"])
        .with_columns(token=pl.Series(vocab.lookup.keys()))
        .with_columns(
            type=pl.col("token").map_elements(
                token_type,
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
        symbol="type",
        width=650,
        title="Token embedding",
        hover_name="token",
        color_discrete_sequence=colors[1:],
    )

    for i, m in enumerate(addl_mdls):
        addl_emb = m.get_input_embeddings()(t_arange(len(vocab)))
        addl_proj = projector.transform(addl_emb.detach())
        addl_df = (
            pl.from_numpy(data=addl_proj, schema=["dim1", "dim2"])
            .with_columns(token=pl.Series(vocab.lookup.keys()))
            .with_columns(
                type=pl.col("token").map_elements(
                    token_type,
                    return_dtype=pl.String,
                    skip_nulls=False,
                )
            )
        )

        addl_fig = go.Scatter(
            x=addl_df["dim1"],
            y=addl_df["dim2"],
            mode="markers",
            marker=dict(
                size=4,
                color=("black", "grey")[i % 2],
                symbol=("x", "cross", "square", "diamond", "circle")[i % 5],
            ),
            text=addl_df["type"],
            hoverinfo="text",
            name=addl_mdls_loc[i].stem.split("-")[-1],
        )

        fig.add_trace(addl_fig)

    fig.update_layout(
        template="plotly_white",
        font_family="CMU Serif",
    )

    # fig.write_html(
    #     out_dir.joinpath("embedding_vis-{m}.html".format(m=ref_mdl_loc.stem))
    # )
    fig.write_image(
        out_dir.joinpath("embedding_vis-{m}.pdf".format(m=ref_mdl_loc.stem))
    )

    """
    quantile embeddings only
    """

    # size: vocab × emb_dim
    emb = ref_mdl.get_input_embeddings()(t_arange(10))
    proj = projector.fit_transform(emb.detach().numpy())
    if projector_type == "PCA":
        logger.info(f"{projector.explained_variance_ratio_=}")

    fig = px.scatter(
        pl.from_numpy(data=proj, schema=["dim1", "dim2"]).with_columns(
            token=pl.Series(list(vocab.lookup.keys())[:10])
        ),
        x="dim1",
        y="dim2",
        color="token",
        symbol="token",
        width=650,
        title="Quantile embedding",
        hover_name="token",
        color_discrete_sequence=colors[1:],
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
                    token_type,
                    return_dtype=pl.String,
                    skip_nulls=False,
                )
            )
        )

        addl_fig = go.Scatter(
            x=addl_df["dim1"],
            y=addl_df["dim2"],
            mode="markers",
            marker=dict(
                size=4,
                color=("black", "grey")[i % 2],
                symbol=("x", "cross", "square", "diamond", "circle")[i % 5],
            ),
            text=addl_df["type"],
            hoverinfo="text",
            name=addl_mdls_loc[i].stem.split("-")[-1],
        )

        fig.add_trace(addl_fig)

    fig.update_layout(
        template="plotly_white",
        font_family="CMU Serif",
    )

    # fig.write_html(out_dir.joinpath("embedding_q-{m}.html".format(m=ref_mdl_loc.stem)))
    fig.write_image(out_dir.joinpath("embedding_q-{m}.pdf".format(m=ref_mdl_loc.stem)))


if __name__ == "__main__":
    fi.Fire(main)
