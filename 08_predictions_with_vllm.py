#!/usr/bin/env python3

"""
generate timeline completions (from just under 24h) from each test sequence;
N.B.: we move the padding from the right (for training) to the left (so that
the most recent context is the timeline and the padding comes beforehand)
"""

import pathlib
import pickle

import torch as t
from datasets import load_dataset
from vllm import LLM, SamplingParams

from vocabulary import Vocabulary

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

data_version = "day_stays_qc_first_24h"
model_version = "smaller-lr-search"  # "small"
model_loc = hm.joinpath("clif-mdls", model_version, "run-2", "checkpoint-18000")

k = 10_000
n_samp = 20
top_p = 0.9

# load and prep data
splits = ("train", "val", "test")
data_dirs = {
    s: hm.joinpath("clif-data", f"{data_version}-tokenized", s) for s in splits
}

vocab = Vocabulary(is_training=False).load(data_dirs["train"].joinpath("vocab.gzip"))


def rt_padding_to_left(t_rt, tk: int = vocab("PAD")):
    i = t.argmax((t_rt == tk).int()).item()
    return t.concat([t.full((t_rt.shape[0] - i,), tk), t_rt[:i]])


s = "test"
dataset = (
    load_dataset(
        "parquet",
        data_files={s: str(data_dirs[s].joinpath("tokens_timelines.parquet"))},
    )
    .map(
        remove_columns=[
            "hospitalization_id",
            "tokens",
            "times",
            "first_fail_or_0",
            "valid_length",
            "seq_len",
        ],
    )
    .with_format("torch")
    .map(
        lambda x: {"input_ids": rt_padding_to_left(x["padded"])},
        remove_columns=["padded"],
    )
)

# load and prep model
model = LLM(
    model=str(model_loc),
    trust_remote_code=True,
    skip_tokenizer_init=True,
)

for rep in range(n_samp):
    sampling_params = SamplingParams(
        top_p=top_p,
        max_tokens=k,
        n=1,
        stop_token_ids=[
            vocab("TL_END"),
            vocab("PAD"),
            vocab("TRUNC"),
            vocab("expired"),
        ],
        detokenize=False,
        seed=rep,
    )
    outp = model.generate(
        prompt_token_ids=dataset[s]["input_ids"].tolist(),
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    response_list = [list(op.outputs[0].token_ids) for op in outp]

    with open(
        data_dirs[s].joinpath(
            f"responses_k{k}_rep_{rep}_of_{n_samp}-{model_version}.pkl"
        ),
        "wb",
    ) as fp:
        pickle.dump(response_list, fp)
