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

data_version = "day_stays_qc_first_24h"
k = 1000
n_samp = 20
top_p = 0.9
model_id = "mdl-day_stays_qc-small-2025-02-05T19:20:52-06:00"
model_version = "small"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

# load and prep data
splits = ("train", "val", "test")
data_dirs = {
    s: hm.joinpath("clif-data", f"{data_version}-tokenized", s) for s in splits
}

vocab = Vocabulary().load(data_dirs["train"].joinpath("vocab.gzip"))
model_dir = hm.joinpath("clif-mdls", model_version)


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
    model=str(model_dir.joinpath(model_id)),
    trust_remote_code=True,
    skip_tokenizer_init=True,
)

for rep in range(n_samp):
    sampling_params = SamplingParams(
        top_p=top_p,
        max_tokens=k,
        n=1,
        stop_token_ids=[vocab("TL_END"), vocab("PAD"), vocab("TRUNC")],
        detokenize=False,
        seed=rep,
    )
    outp = model.generate(
        prompt_token_ids=dataset[s]["input_ids"].tolist(),
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    response_list = [[list(x.token_ids) for x in op.outputs] for op in outp]

    with open(
        data_dirs[s].joinpath(f"responses_k{k}_rep_{rep}_of_{n_samp}.pkl"), "wb"
    ) as fp:
        pickle.dump(response_list, fp)
