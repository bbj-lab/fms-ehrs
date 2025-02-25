#!/usr/bin/env python3

"""
generate timeline completions (from just under 24h) from each test sequence;
N.B.: we move the padding from the right (for training) to the left (so that
the most recent context is the timeline and the padding comes beforehand)
"""

import argparse
import logging
import os
import pathlib
import pickle

import torch as t
from datasets import load_dataset
from vllm import LLM, SamplingParams

from logger import get_logger
from vocabulary import Vocabulary

# suppress tons of: WARNING XX-XX XX:XX:XX preprocess.py:58]
# Using None for EOS token id because tokenizer is not initialized
logging.getLogger("vllm").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--rep", type=int, required=False, default=0)
args, _ = parser.parse_known_args()
rep = args.rep

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/").expanduser()

data_version = "day_stays_qc_first_24h"
model_loc = hm.joinpath(
    "clif-mdls-archive", "medium-packing-tuning-57164794-run2-ckpt-7000"
)

k = 25_000
n_samp = 1
top_p = 0.95

if os.getenv("RANK", "0") == "0":
    logger = get_logger()
    logger.info("running {}".format(__file__))
    logger.log_env()
    logger.info(f"{rep=}")
    logger.info(f"{data_version=}")
    logger.info(f"{model_loc.stem=}")
    logger.info(f"{model_loc=}")
    logger.info(f"{k=}")
    logger.info(f"{n_samp=}")
    logger.info(f"{top_p=}")


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

context_length = dataset["test"].select(range(1))["input_ids"].shape[1]


"""
If `max_model_len < max_tokens + context_length`, 
generation will stop at max_model_len - context_length 
without warning or error...
"""

model = LLM(
    model=str(model_loc),
    skip_tokenizer_init=True,
    max_model_len=k + context_length,
    task="generate",
    max_seq_len_to_capture=k,
)

sampling_params = SamplingParams(
    top_p=top_p,
    max_tokens=k,
    n=1,
    stop_token_ids=[
        vocab("TL_END"),
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
    data_dirs[s].joinpath(f"responses_k{k}_rep_{rep}_of_{n_samp}-{model_loc.stem}.pkl"),
    "wb",
) as fp:
    pickle.dump(response_list, fp)
