#!/usr/bin/env python3

"""
grab the sequence of logits from the test set
"""

import argparse
import collections
import os
import pathlib
import typing

import numpy as np
import torch as t
from datasets import concatenate_datasets, load_dataset
from plotly import express as px
from plotly import graph_objects as go
from plotly import io as pio
from transformers import AutoModelForCausalLM

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import fix_perms, set_perms
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
    default=[
        "24640534",  # cf. Fig. 2
        "26886976",  # Fig. 3
        "29022625",  # Fig. 4
        "20606203",
        "29298288",
        "28910506",
        "28812737",
        "20606203",
        "29866426",
    ],
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--final_layer", action="store_true")
parser.add_argument("--log_scale", action="store_true")
parser.add_argument("--max_len", type=int, default=42)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir),
)
out_dir.mkdir(parents=True, exist_ok=True)
fix_perms(out_dir)

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
        use_cache=True,
    )
attns = np.stack(
    [_.cpu() for _ in x.attentions]
)  # n_layers × batch_size × num_heads × sequence_length × sequence_length
vals = np.stack(
    [_[1].cpu() for _ in x.past_key_values]
)  # n_layers × batch_size × num_heads × sequence_length × d_vals

agg_attn = (
    attns[-1].mean(axis=1) if args.final_layer else attns.mean(axis=(0, 2))
)  # now: batch_size × sequence_length × sequence_length
agg_vals = (
    np.linalg.norm(vals[-1], axis=-1, ord=1).mean(axis=1)
    if args.final_layer
    else np.linalg.norm(vals, axis=-1, ord=1).mean(axis=(0, 2))
)  # now: batch_size × sequence_length

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

importances = collections.OrderedDict()
importances["H2O"] = attns.sum(axis=(0, 2, 3))
importances["H2O-VA"] = (
    attns * np.linalg.norm(vals, axis=-1, ord=1, keepdims=True)
).sum(axis=(0, 2, 3))
importances["✂️👐🏻-10"] = (
    attns * (np.tri(vals.shape[3]) - np.tri(vals.shape[3], k=-10))
).sum(axis=(0, 2, 3))
importances["✂️👐🏻-10-VA"] = (
    attns
    * (np.tri(vals.shape[3]) - np.tri(vals.shape[3], k=-10))
    * np.linalg.norm(vals, axis=-1, ord=1, keepdims=True)
).sum(axis=(0, 2, 3))
importances["✂️👐🏻-20"] = (
    attns * (np.tri(vals.shape[3]) - np.tri(vals.shape[3], k=-20))
).sum(axis=(0, 2, 3))
importances["✂️👐🏻-20-VA"] = (
    attns
    * (np.tri(vals.shape[3]) - np.tri(vals.shape[3], k=-20))
    * np.linalg.norm(vals, axis=-1, ord=1, keepdims=True)
).sum(axis=(0, 2, 3))

assert np.all(importances["H2O"] + 0.01 >= importances["✂️👐🏻-20"])
assert np.all(importances["✂️👐🏻-20"] + 0.01 >= importances["✂️👐🏻-10"])

# importances = np.nanmean(
#     np.where(
#         np.isfinite(
#             a := (np.log if args.log_scale else lambda _: _)(
#                 agg_attn[:, : args.max_len, : args.max_len]
#             )
#         ),
#         a,
#         np.nan,
#     ),
#     axis=1,
# )
# importances[:, -10:] = importances.mean(
#     axis=-1, keepdims=True
# )  # this doesn't work well on things that don't have a history
q95 = np.quantile(importances["H2O-VA"][:, : args.max_len], 0.9, axis=1)

for i in range(len(agg_attn)):
    fig = go.Figure(
        data=go.Heatmap(
            z=(np.log if args.log_scale else lambda _: _)(
                agg_attn[i, : args.max_len, : args.max_len]
            ),
            x=list(range(args.max_len)),
            y=list(range(args.max_len)),
            colorscale=px.colors.sequential.Viridis,
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
            tickvals=list(range(args.max_len)),
            ticktext=[
                x if importances["H2O-VA"][i][j] < q95[i] else "<b>{}</b>".format(x)
                for j, x in enumerate(selected_decoded[i, : args.max_len])
            ],
            showticklabels=True,
            tickangle=-45,
            tickfont=dict(size=36),
            ticklabelstandoff=-100,
            side="bottom" if args.log_scale else "top",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            autorange="reversed",
            tickvals=list(range(args.max_len)),
            ticktext=selected_decoded[i, : args.max_len],
            tickfont=dict(size=36),
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
