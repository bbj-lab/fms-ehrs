#!/usr/bin/env python3

"""
generate timeline completions (from just under 24h) from each test sequence
"""

import argparse
import gzip
import os
import pathlib
import pickle
import typing

import polars as pl
from vllm import LLM, SamplingParams, TokensPrompt

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import fix_perms
from fms_ehrs.framework.vocabulary import Vocabulary

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument(
    "--data_version",
    type=str,
    default="W++_first_24h_llama-med-60358922_1-hp-W++_none_10pct_ppy",
)
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="../../mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--max_len", type=int, default=10_000)
parser.add_argument("--n_samp", type=int, default=20)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

df_test = pl.scan_parquet(
    data_dir / f"{args.data_version}-tokenized" / "test" / "tokens_timelines.parquet"
)

vocab = Vocabulary().load(
    data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip"
)

# load and prep model
model = LLM(model=str(model_loc), skip_tokenizer_init=True)

sampling_params = SamplingParams(
    max_tokens=args.max_len,
    n=args.n_samp,
    stop_token_ids=[vocab("TL_END"), vocab("PAD"), vocab("TRUNC")],
    detokenize=False,
    seed=0,
)

test_token_list = df_test.select("tokens").collect().to_series().to_list()

"""sample completions for 24h-cutoff test set"""
outp = model.generate(
    prompts=[TokensPrompt(prompt_token_ids=x) for x in test_token_list],
    sampling_params=sampling_params,
    use_tqdm=True,
)

res = [[out.token_ids for out in op.outputs] for op in outp]

with gzip.open(
    data_dir / f"{args.data_version}-tokenized" / "test" / "gen_preds.pkl.gz", "w+"
) as f:
    pickle.dump(res, f)
    fix_perms(f)

logger.info("---fin")
