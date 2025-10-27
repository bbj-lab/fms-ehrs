#!/usr/bin/env python3

"""
extract attention maps and values (from queries, keys, & values);
consider token importance metrics as described in Guo, Kamigaito, Watanabe's
"Attention score is not all you need for token importance indicator in KV cache
reduction: Value also matters" (Empirical Methods in Natural Language Processing, 2024)
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
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import fix_perms, set_perms
from fms_ehrs.framework.vocabulary import Vocabulary
from fms_ehrs.framework.plotting import imshow_text

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
parser.add_argument("--data_version", type=str, default="W++")
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
        # "20606203",
        # "29298288",
        # "28910506",
        # "28812737",
        # "20606203",
        # "29866426",
    ],
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--final_layer", action="store_true")
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

metrics = collections.OrderedDict()
metrics["value-norms"] = np.linalg.norm(vals, axis=-1, ord=1).sum(axis=(0, 2))
metrics["H2O"] = attns.sum(axis=(0, 2, 3))
metrics["H2O-VA"] = (attns * np.linalg.norm(vals, axis=-1, ord=1, keepdims=True)).sum(
    axis=(0, 2, 3)
)
metrics["SH-10"] = (attns * (np.tri(vals.shape[3]) - np.tri(vals.shape[3], k=-10))).sum(
    axis=(0, 2, 3)
)
metrics["SH-10-VA"] = (
    attns
    * (np.tri(vals.shape[3]) - np.tri(vals.shape[3], k=-10))
    * np.linalg.norm(vals, axis=-1, ord=1, keepdims=True)
).sum(axis=(0, 2, 3))
metrics["SH-20"] = (attns * (np.tri(vals.shape[3]) - np.tri(vals.shape[3], k=-20))).sum(
    axis=(0, 2, 3)
)
metrics["SH-20-VA"] = (
    attns
    * (np.tri(vals.shape[3]) - np.tri(vals.shape[3], k=-20))
    * np.linalg.norm(vals, axis=-1, ord=1, keepdims=True)
).sum(axis=(0, 2, 3))

metrics_normalized = {
    k: v[:, : args.max_len] / v[:, : args.max_len].sum(axis=-1, keepdims=True)
    for k, v in metrics.items()
}

for i in range(len(agg_attn)):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    fig.add_trace(
        go.Heatmap(
            z=np.log(agg_attn[i, : args.max_len, : args.max_len]),
            x=list(range(args.max_len)),
            y=list(range(args.max_len)),
            colorscale=px.colors.sequential.Viridis,
            reversescale=False,
            showscale=True,
            zsmooth=False,
            xgap=1,
            ygap=1,
            name="top",
            colorbar=dict(title="attentions", len=0.5, y=0.775),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=np.vstack([v[i, : args.max_len] for v in metrics_normalized.values()]),
            x=list(range(args.max_len)),
            y=list(range(args.max_len)),
            colorscale=px.colors.sequential.Viridis,
            reversescale=False,
            showscale=True,
            zsmooth=False,
            xgap=1,
            ygap=1,
            name="bottom",
            colorbar=dict(title="values & metrics", len=0.5, orientation="h", y=0.1),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickvals=list(range(args.max_len)),
            ticktext=selected_decoded[i, : args.max_len],
            showticklabels=True,
            tickangle=-90,
            tickfont=dict(size=20),
            side="bottom",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            autorange="reversed",
            tickvals=list(range(args.max_len)),
            ticktext=selected_decoded[i, : args.max_len],
            tickfont=dict(size=20),
            # scaleanchor="x",
        ),
        height=2000,
        width=2000,
        font_family="CMU Serif, Times New Roman, serif",
        xaxis_scaleanchor="y",
        plot_bgcolor="white",
        template="plotly_white",
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=True,
        autorange="reversed",
        tickvals=list(np.arange(len(metrics.keys())) + 0.5),
        ticktext=list(metrics.keys()),
        tickfont=dict(size=20),
        scaleanchor="x",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=2,
        col=1,
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

args.max_len = 210
n_cols = 6
n_rows = args.max_len // n_cols
max_len = n_rows * n_cols
height = (700 * n_rows) // 42

for k, v in metrics.items():
    for i in range(len(agg_attn)):
        imshow_text(
            values=v[i][:max_len].reshape((-1, n_cols)),
            text=selected_decoded[i, :max_len].reshape((-1, n_cols)),
            savepath=out_dir.joinpath(
                "tls-{}-{}.pdf".format(selected_data["hospitalization_id"][i], k)
            ),
            autosize=False,
            zmin=v[:, :max_len].min(),
            zmax=v[:, 8:max_len].max(),
            height=height,
            width=1000,
            margin=dict(l=0, r=0, t=0, b=0),
        )

logger.info("---fin")
