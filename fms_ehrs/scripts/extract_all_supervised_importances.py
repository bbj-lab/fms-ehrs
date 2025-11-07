#!/usr/bin/env python3

"""
extract token saliency using fine-tuned models
"""

import argparse
import pathlib

import datasets as ds
import numpy as np
import torch as t
import tqdm as tq
from captum.attr import NoiseTunnel, Saliency
from transformers import AutoModelForSequenceClassification

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.util import rt_padding_to_left
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++-sft-mort",
)
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--data_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--outcome",
    choices=[
        "same_admission_death",
        "long_length_of_stay",
        "icu_admission",
        "imv_event",
    ],
    default="same_admission_death",
)
parser.add_argument("--noise_tunnel", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")


model_loc, data_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.model_loc, args.data_dir)
)

splits = ("train", "val", "test")
data_dirs = {s: data_dir.joinpath(f"{args.data_version}-tokenized", s) for s in splits}
vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))

dataset = (
    ds.load_dataset(
        "parquet",
        data_files={
            s: str(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
            for s in ("test",)
        },
    )
    .map(
        lambda x: {"same_admission_death_24h": False, "long_length_of_stay_24h": False}
    )
    .select_columns(["padded", args.outcome, f"{args.outcome}_24h"])
    .with_format("torch")
    .map(
        lambda x: {
            "input_ids": rt_padding_to_left(x["padded"], vocab("PAD")),
            "label": x[args.outcome],
        },
        remove_columns=["padded", args.outcome],
    )
)

model = AutoModelForSequenceClassification.from_pretrained(model_loc).to("cuda")


def forward_func(ems):
    return t.softmax(model(inputs_embeds=ems.to("cuda")).logits, dim=1)[:, 1]


s = Saliency(forward_func)
if args.noise_tunnel:
    nt = NoiseTunnel(s)

input_ids = t.stack(list(dataset["test"]["input_ids"]))
saliency = np.zeros(shape=input_ids.shape, dtype=np.float32)
batches = t.split(t.arange(saliency.shape[0]), args.batch_size)

for batch_idx in tq.tqdm(batches):
    batch_embeds = model.get_input_embeddings()(input_ids[batch_idx].to("cuda"))
    saliency[batch_idx] = (
        t.norm(
            (
                nt.attribute(
                    batch_embeds,
                    nt_type="smoothgrad",
                    nt_samples=5,
                    nt_samples_batch_size=1,
                )
                if args.noise_tunnel
                else s.attribute(inputs=batch_embeds)
            ),
            dim=-1,
        )
        .cpu()
        .detach()
        .numpy()
        .astype(np.float32)
    )

set_perms(np.save, compress=True)(
    data_dirs["test"].joinpath(
        ("smoothgrad-{mdl}.npy" if args.noise_tunnel else "saliency-{mdl}.npy").format(
            mdl=model_loc.stem
        )
    ),
    saliency,
)

logger.info("---fin")
