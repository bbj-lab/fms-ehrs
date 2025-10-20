#!/usr/bin/env python3

"""
extract attention maps and values (from queries, keys, & values)
*for a smallish (1 batch) sample of hospitalizations*;
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
from fms_ehrs.framework.plotting import imshow_text
from fms_ehrs.framework.storage import fix_perms, set_perms
from fms_ehrs.framework.util import agg_str2fn, token_importance0
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
parser.add_argument("--max_len", type=int, default=42)
parser.add_argument(
    "--agg_fns", nargs="+", default=["sum", "mean", "max", "median", "Q90", "Q95"]
)
parser.add_argument("--drop_labels", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc, out_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir, args.model_loc, args.out_dir),
)
out_dir.mkdir(parents=True, exist_ok=True)
fix_perms(out_dir)

t.cuda.set_device(device := t.device("cuda:0"))

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
    model_loc, attn_implementation="eager"
)  # in eval mode by default
d = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = model.config.head_dim
w_out = model.model.layers[0].self_attn.o_proj.weight  # d × (num_heads * head_dim)
assert w_out.shape[-1] == num_heads * head_dim
model = model.to(device)

with t.inference_mode():
    x = model.forward(
        input_ids=selected_data["input_ids"][: len(selected_data)].to(device),
        output_attentions=True,
        use_cache=True,
    )
attns = np.stack(
    [_[0].cpu() for _ in x.attentions]
)  # batch_size × num_heads × sequence_length × sequence_length
vals = np.stack(
    [_[1][0].cpu() for _ in x.past_key_values]
)  # batch_size × num_head_groups × sequence_length × head_dim

# Llama3 uses grouped query attention
# see, https://www.ibm.com/think/topics/grouped-query-attention
# the following function does ungrouping for us:
# https://github.com/meta-llama/llama/blob/4d92db8a1db6c7f663252bf3477d2c4b8bad2385/llama/model.py#L77
if (n_per_gp := attns.shape[1] // vals.shape[1]) > 1:
    vals = np.repeat(
        vals, repeats=n_per_gp, axis=1
    )  # now, batch_size × num_heads × sequence_length × d_vals

normed_wts = np.stack(
    [
        np.linalg.norm(x, ord=1, axis=-1)
        for x in np.split(w_out.detach().cpu().numpy().T, num_heads)
    ]
).reshape((1, num_heads, 1, head_dim))


normed_vals = np.linalg.norm(
    vals * normed_wts, axis=-1, ord=1, keepdims=True
)  # batch_size × num_heads × sequence_length × 1

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
)  # batch_size × sequence_length


for agg_fn_str in args.agg_fns:
    agg_attn = agg_str2fn(agg_fn_str)(
        attns, axis=1
    )  # now: batch_size × sequence_length × sequence_length
    metrics = collections.OrderedDict()
    metrics["value-norms"] = agg_str2fn(agg_fn_str)(normed_vals, axis=(1, -1))
    metrics["H2O"] = token_importance0(attns, aggregation=agg_fn_str)
    metrics["H2O-VA"] = token_importance0(
        attns, normed_values=normed_vals, aggregation=agg_fn_str
    )
    metrics["SH-100"] = token_importance0(attns, window=100, aggregation=agg_fn_str)
    metrics["SH-100-VA"] = token_importance0(
        attns, normed_values=normed_vals, window=100, aggregation=agg_fn_str
    )
    # metrics["SH-20"] = token_importance0(attns, window=20, aggregation=agg_fn_str)
    # metrics["SH-20-VA"] = token_importance0(
    #     attns, normed_values=normed_vals, window=20, aggregation=agg_fn_str
    # )

    metrics_normalized = {
        k: v[:, : args.max_len] / v[:, : args.max_len].sum(axis=-1, keepdims=True)
        for k, v in metrics.items()
    }

    for i in range(len(selected_data)):
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
                z=np.vstack(
                    [v[i, : args.max_len] for v in metrics_normalized.values()]
                ),
                x=list(range(args.max_len)),
                y=list(range(args.max_len)),
                colorscale=px.colors.sequential.Viridis,
                reversescale=False,
                showscale=True,
                zsmooth=False,
                xgap=1,
                ygap=1,
                name="bottom",
                colorbar=dict(
                    title="values & metrics", len=0.5, orientation="h", y=0.1
                ),
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
                showticklabels=not args.drop_labels,
                tickangle=-90,
                tickfont=dict(size=20),
                side="bottom",
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=not args.drop_labels,
                autorange="reversed",
                tickvals=list(range(args.max_len)),
                ticktext=selected_decoded[i, : args.max_len],
                tickfont=dict(size=20),
                scaleanchor="x" if args.drop_labels else None,
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
            showgrid=False, zeroline=False, showticklabels=False, row=2, col=1
        )
        set_perms(fig.write_image)(
            pathlib.Path(
                out_dir.joinpath(
                    "attn-{sid}-{agg}-{dv}-{mv}.pdf".format(
                        sid=selected_data["hospitalization_id"][i],
                        agg=agg_fn_str,
                        dv=args.data_version,
                        mv=model_loc.stem,
                    )
                )
            )
            .expanduser()
            .resolve()
        )

    n_cols = 6
    n_rows = args.max_len // n_cols
    max_len = n_rows * n_cols
    height = (700 * n_rows) // 42

    for k, v in metrics_normalized.items():
        for i in range(len(selected_data)):
            imshow_text(
                values=v[i][:max_len].reshape((-1, n_cols)),
                text=selected_decoded[i, :max_len].reshape((-1, n_cols)),
                savepath=out_dir.joinpath(
                    "tls-{sid}-{met}-{agg}-{dv}-{mv}.pdf".format(
                        sid=selected_data["hospitalization_id"][i],
                        met=k,
                        agg=agg_fn_str,
                        dv=args.data_version,
                        mv=model_loc.stem,
                    )
                ),
                autosize=False,
                zmin=v[:, :max_len].min(),
                zmax=v[:, 8:max_len].max(),
                height=height,
                width=1000,
                margin=dict(l=0, r=0, t=0, b=0),
            )

logger.info("---fin")
