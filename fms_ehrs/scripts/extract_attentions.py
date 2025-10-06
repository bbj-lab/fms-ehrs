#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import collections
import os
import pathlib
import typing

import torch as t
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM
import numpy as np
from plotly import express as px
from plotly import graph_objects as go
from plotly import io as pio

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike
Dictlike: typing.TypeAlias = collections.OrderedDict | dict

try:
    pio.defaults.mathjax = None
except AttributeError:
    pio.kaleido.scope.mathjax = None

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument(
    "--ids",
    type=str,
    nargs="*",
    default=["20606203", "29298288", "28910506", "28812737", "20606203", "29866426"],
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--final_layer", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir),
)

rank = 0
device = t.device(f"cuda:{rank}")
t.cuda.set_device(device)

# load and prep data
splits = ("train", "val", "test")
data_dirs = dict()
for s in splits:
    data_dirs[s] = data_dir.joinpath(f"{args.data_version}-tokenized", s)

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

selected_data = concatenate_datasets(
    list(
        load_dataset(
            "parquet",
            data_files={
                s: str(data_dirs[s].joinpath("tokens_timelines.parquet"))
                for s in splits
            },
        )
        .map(
            lambda batch: {"input_ids": batch["padded"]},
            batched=True,
            remove_columns=["tokens", "times", "seq_len"],
        )
        .filter(lambda x: x["hospitalization_id"] in args.ids)
        .with_format("torch")
        .values()
    )
)

# load and prep model
model = AutoModelForCausalLM.from_pretrained(
    model_loc, attn_implementation="eager", output_attentions=True
)  # in eval mode by default
d = model.config.hidden_size
model = model.to(device)

with t.inference_mode():
    x = model.forward(
        input_ids=selected_data["input_ids"][: len(selected_data)].to(device),
        output_attentions=True,
    )
attns = (
    x.attentions
)  # a tuple with n_layers: batch_size × num_heads × sequence_length × sequence_length arrays

agg_attn = (
    (attns[-1].cpu().numpy().mean(axis=1))
    if args.final_layer
    else np.stack(tuple(map(lambda x: x.cpu(), attns))).mean(axis=(0, 2))
)  # now: batch_size × sequence_length × sequence_length

selected_decoded = np.array(
    [
        [
            (
                (d if len(d) <= 23 else f"{d[:13]}..{d[-7:]}")
                if (d := vocab.reverse[t]) is not None
                else "None"
            )
            for t in x
        ]
        for x in selected_data["input_ids"][: len(selected_data)].numpy()
    ]
)

max_len = 100
for i in range(len(agg_attn)):
    fig = go.Figure(
        data=go.Heatmap(
            z=agg_attn[i, :max_len, :max_len],
            x=list(range(max_len)),
            y=list(range(max_len)),
            colorscale=px.colors.sequential.Viridis[4:],
            reversescale=False,
            showscale=True,
            zsmooth=False,
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickvals=list(range(max_len)),
            ticktext=selected_decoded[i, :max_len],
            showticklabels=True,
            tickangle=-45,
            tickfont=dict(size=12),
            # ticklabelshift=48,
            ticklabelstandoff=-48,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            autorange="reversed",
            tickvals=list(range(max_len)),
            ticktext=selected_decoded[i, :max_len],
            tickfont=dict(size=12),
        ),
        height=3000,
        width=3000,
        font_family="CMU Serif, Times New Roman, serif",
        xaxis_scaleanchor="y",
        plot_bgcolor="white",
        template="plotly_white",
    )
    set_perms(fig.write_image)(
        pathlib.Path(
            out_dir.joinpath(
                "attn-{}-{}.pdf".format(
                    selected_data["hospitalization_id"][i],
                    "final" if args.final_layer else "all",
                )
            )
        )
        .expanduser()
        .resolve()
    )

logger.info("---fin")
