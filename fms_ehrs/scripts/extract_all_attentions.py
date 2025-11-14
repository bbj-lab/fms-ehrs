#!/usr/bin/env python3

"""
process importance metrics for all timelines in batches
"""

import argparse
import os
import pathlib
import typing

import numpy as np
import torch as t
import torch.distributed as dist
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike

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
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--batch_num_start", type=int, default=None)
parser.add_argument("--batch_num_end", type=int, default=None)
parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
parser.add_argument(
    "--metrics",
    nargs="*",
    default=[
        "h2o-mean",
        "h2o-mean_log",
        "h2o-va-mean",
        "h2o-va-mean_log",
        "scissorhands-10",
        "scissorhands-20",
        "scissorhands-va-10",
        "scissorhands-va-20",
        "rollout-mean",
        "rollout-mean_log",
        "h2o-normed-mean",
        "h2o-normed-mean_log",
    ],
)
parser.add_argument("--use_jax", action="store_true")
args, unknowns = parser.parse_known_args()

if args.use_jax:
    from fms_ehrs.framework.util_jax import token_importance
else:
    from fms_ehrs.framework.util import token_importance

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

# prepare parallelism
is_parallel = t.cuda.device_count() > 1
if is_parallel:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
else:
    rank = 0
device = t.device(f"cuda:{rank}")
t.cuda.set_device(device)

# load and prep data
splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
stop_tokens = t.tensor([vocab("TRUNC"), vocab("TL_END")])

# load and prep model
model = AutoModelForCausalLM.from_pretrained(
    model_loc, attn_implementation="eager"
)  # in eval mode by default

d = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = model.config.head_dim

w_out = model.model.layers[0].self_attn.o_proj.weight  # d × (num_heads * head_dim)
wts = np.stack(
    np.split(w_out.detach().cpu().numpy().T, num_heads)
)  # num_heads × d_vals × d

model = model.to(device)
if is_parallel:
    model = t.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

dataset = (
    load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines.parquet")) for s in splits
        },
    )
    .map(lambda batch: {"input_ids": batch["padded"]}, batched=True)
    .with_format("torch")
)

for s in args.splits:
    n = dataset[s].num_rows
    tl_len = len(dataset[s].select(range(1))["input_ids"][0])
    metrics = {k: np.zeros(shape=(n, tl_len), dtype=np.float32) for k in args.metrics}
    batches = t.split(t.arange(n), args.batch_size)
    logger.warning(f"For split {s=}, {len(batches)=} in total are required.")
    for batch_num, batch_idx in tqdm(enumerate(batches)):
        if args.batch_num_start is not None and batch_num < args.batch_num_start:
            continue
        if args.batch_num_end is not None and batch_num >= args.batch_num_end:
            continue
        batch = dataset[s]["input_ids"][batch_idx].to(device)
        with t.inference_mode():
            x = model.forward(input_ids=batch, output_attentions=True, use_cache=True)
        attns = np.stack(
            [_.cpu() for _ in x.attentions]
        )  # n_layers × batch_size × num_heads × sequence_length × sequence_length
        vals = np.stack(
            [_[1].cpu() for _ in x.past_key_values]
        )  # n_layers × batch_size × num_heads × sequence_length × d_vals
        # Llama3 uses grouped query attention
        # see, https://www.ibm.com/think/topics/grouped-query-attention
        n_groups = attns.shape[2] // vals.shape[2]
        if n_groups > 1:
            vals = np.repeat(vals, repeats=n_groups, axis=2)
        first_stop_idx = t.argmax(
            t.isin(batch, stop_tokens.to(device)).int(), dim=1, keepdim=True
        )  # or 0 if no stop token
        for met in args.metrics:
            match met:
                case "h2o-mean" | "h2o-mean_log":
                    metrics[met][batch_idx] = token_importance(
                        attentions=attns, aggregation=met.split("-")[-1]
                    ).astype(np.float32)
                case "h2o-va-mean" | "h2o-va-mean_log":
                    metrics[met][batch_idx] = token_importance(
                        attentions=attns, values=vals, aggregation=met.split("-")[-1]
                    ).astype(np.float32)
                case "scissorhands-10" | "scissorhands-20":
                    metrics[met][batch_idx] = token_importance(
                        attentions=attns,
                        window=int(met.split("-")[-1]),
                        aggregation="mean",
                    ).astype(np.float32)
                case "scissorhands-va-10" | "scissorhands-va-20":
                    metrics[met][batch_idx] = token_importance(
                        attentions=attns,
                        values=vals,
                        window=int(met.split("-")[-1]),
                        aggregation="mean",
                    ).astype(np.float32)
                case "rollout-mean" | "rollout-mean_log":
                    metrics[met][batch_idx] = token_importance(
                        attentions=attns, rollout=True, aggregation=met.split("-")[-1]
                    ).astype(np.float32)
                case "h2o-normed-mean" | "h2o-normed-mean_log":
                    # ||af|| = |a|*||f||
                    alpha_fs_normed = (
                        np.abs(attns)
                        * np.expand_dims(
                            np.linalg.norm(np.matmul(vals, wts), axis=-1), axis=-1
                        )
                    )  # n_layers × batch_size × num_heads × sequence_length × sequence_length
                    metrics[met][batch_idx] = token_importance(
                        attentions=alpha_fs_normed, aggregation=met.split("-")[-1]
                    ).astype(np.float32)
            for i, j in enumerate(first_stop_idx.cpu().numpy().ravel()):
                if j > 0:
                    metrics[met][batch_idx[i], j + 1 :] = np.nan

    for met in args.metrics:
        set_perms(np.save, compress=True)(
            data_dirs[s].joinpath(
                "importance-{met}-{mdl}{sn}{en}.npy".format(
                    met=met,
                    mdl=model_loc.stem,
                    sn=(
                        "-s" + str(ns).zfill(4)
                        if (ns := args.batch_num_start) is not None
                        else ""
                    ),
                    en=(
                        "-e" + str(ne).zfill(4)
                        if (ne := args.batch_num_end) is not None
                        else ""
                    ),
                )
            ),
            metrics[met],
        )


logger.info("---fin")
