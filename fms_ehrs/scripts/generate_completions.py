#!/usr/bin/env python3

"""
generate timeline completions (from just under 24h) from each test sequence
"""

import argparse
import os
import pathlib
import typing

import polars as pl
from vllm import LLM, SamplingParams, TokensPrompt

from fms_ehrs.framework.logger import get_logger
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
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

k = 10_000
n_samp = 5
top_p = 0.9


df_24 = pl.scan_parquet(
    data_dir
    / f"{args.data_version}_first_24h-tokenized"
    / "test"
    / "tokens_timelines.parquet"
)

df = df_24.join(
    pl.scan_parquet(
        data_dir
        / f"{args.data_version}-tokenized"
        / "test"
        / "tokens_timelines.parquet"
    ),
    on="hospitalization_id",
    validate="1:1",
    suffix="_full",
    how="left",
)

vocab = Vocabulary().load(
    data_dir / f"{args.data_version}_first_24h-tokenized" / "train" / "vocab.gzip"
)

s = "test"

# load and prep model
model = LLM(model=str(model_loc), skip_tokenizer_init=True)

sampling_params = SamplingParams(
    max_tokens=k,
    n=n_samp,
    stop_token_ids=[vocab("TL_END"), vocab("PAD"), vocab("TRUNC")],
    detokenize=False,
    seed=0,
)

test_token_list = df_24.select("tokens").collect().to_series().to_list()
true_completions = df.select("tokens_full").collect().to_series().to_list()

"""sample completions for 24h-cutoff test set"""
n_ex = 20
outp = model.generate(
    prompts=[TokensPrompt(prompt_token_ids=x) for x in test_token_list[:n_ex]],
    sampling_params=sampling_params,
    use_tqdm=True,
)

dtkz = lambda x: [vocab.reverse[y] for y in x]

for i in range(n_ex):
    print("=" * 42)
    print("start:", dtkz(test_token_list[i]))
    print("true completion:", dtkz(true_completions[i][len(test_token_list[i]) :]))
    for out in outp[i].outputs:
        print("completion: ", dtkz(out.token_ids))


"""sample random full timelines"""
outp = model.generate(
    prompts=TokensPrompt(prompt_token_ids=[vocab("TL_START")]),
    sampling_params=sampling_params,
    use_tqdm=True,
)

for out in outp[0].outputs:
    print(["TL_START"] + [vocab.reverse[y] for y in out.token_ids])


logger.info("---fin")
