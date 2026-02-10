#!/usr/bin/env python3

"""
generate timeline completions (from just under 24h) from each test sequence
"""

import argparse
import itertools
import os
import pathlib
import typing

import numpy as np
import polars as pl
import tqdm
from vllm import LLM, SamplingParams, TokensPrompt

from fms_ehrs.framework.logger import get_logger, log_classification_metrics
from fms_ehrs.framework.stats import bootstrap_ci
from fms_ehrs.framework.vocabulary import Vocabulary

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=pathlib.Path,
    default="/gpfs/data/bbj-lab/users/burkh4rt/data-mimic"
    if os.uname().nodename.startswith("cri")
    else "/mnt/bbj-lab/users/burkh4rt/data-mimic",
)
parser.add_argument(
    "--data_version",
    type=str,
    default="W++_first_24h_llama-med-60358922_1-hp-W++_none_10pct_ppy",
)
parser.add_argument("--tto_version", type=str, default="W++_first_24h")
parser.add_argument(
    "--model_loc",
    type=pathlib.Path,
    default="/gpfs/data/bbj-lab/users/burkh4rt/mdls-archive/llama-med-60358922_1-hp-W++"
    if os.uname().nodename.startswith("cri")
    else "/mnt/bbj-lab/users/burkh4rt/mdls-archive/llama-med-60358922_1-hp-W++",
)
parser.add_argument("--max_len", type=int, default=100_000)
parser.add_argument("--n_samp", type=int, default=20)
parser.add_argument("--test_size", type=int, default=1_000)
parser.add_argument("--batch_size", type=int, default=32)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, model_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.model_loc)
)

df_test = (
    pl.read_parquet(
        data_dir
        / f"{args.data_version}-tokenized"
        / "test"
        / "tokens_timelines.parquet"
    )
    .sample(n=args.test_size)
    .lazy()
)
test_token_list = df_test.select("tokens").collect().to_series().to_list()

vocab = Vocabulary().load(
    data_dir / f"{args.data_version}-tokenized" / "train" / "vocab.gzip"
)

# load and prep model
model = LLM(model=str(model_loc), skip_tokenizer_init=True, max_logprobs=len(vocab))

check_01 = list()
M0 = list()
M1 = list()

for batch in tqdm.tqdm(itertools.batched(test_token_list, args.batch_size)):
    for op in model.generate(
        prompts=[TokensPrompt(prompt_token_ids=x) for x in batch],
        sampling_params=SamplingParams(
            max_tokens=args.max_len,
             allowed_token_ids=[
                v for k, v in vocab.lookup.items() if not (k in ["PAD", "TRUNC"])
            ],
            n=args.n_samp,
            stop_token_ids=[
                vocab("TL_END"),
                vocab("DSCG_expired"),
            ],
            detokenize=False,
            seed=0,
            logprobs=-1,
        ),
    ):
        logger.info(f"prompt_len={len(batch[0])} total_lens={[len(batch[0]) + len(out.token_ids) for out in op.outputs]}")
        check_01.append([vocab("TL_END") in out.token_ids or vocab("DSCG_expired") in out.token_ids for out in op.outputs])
        M0.append(
            np.mean([vocab("DSCG_expired") in out.token_ids for out in op.outputs])
        )
        M1.append(
            np.mean(
                [
                    np.sum(
                        np.exp(
                            [lp[vocab("DSCG_expired")].logprob for lp in out.logprobs]
                        )
                    )
                    for out in op.outputs
                ]
            )
        )

if not np.array(check_01).all():
    logger.warning(
        "Completion rate for first run: {:.2f}".format(np.array(check_01).mean())
    )

check_2 = list()
check_2_avoid = list()
M2 = list()

for batch in tqdm.tqdm(itertools.batched(test_token_list, args.batch_size)):
    for op in model.generate(
        prompts=[TokensPrompt(prompt_token_ids=x) for x in batch],
        sampling_params=SamplingParams(
            allowed_token_ids=[
                v for k, v in vocab.lookup.items() if not (k in ["PAD", "TRUNC", "DSCG_expired"])
            ],
            max_tokens=args.max_len,
            n=args.n_samp,
            stop_token_ids=[vocab("TL_END")],
            detokenize=False,
            seed=0,
            logprobs=-1,
        ),
    ):
        logger.info(f"prompt_len={len(batch[0])} total_lens={[len(batch[0]) + len(out.token_ids) for out in op.outputs]}")
        check_2.append([vocab("TL_END") in out.token_ids for out in op.outputs])
        M2.append(
            np.mean(
                [
                    1.0
                    - np.prod(
                        1.0
                        - np.exp(
                            [lp[vocab("DSCG_expired")].logprob for lp in out.logprobs]
                        )
                    )
                    for out in op.outputs
                ]
            )
        )

if not np.array(check_2).all():
    logger.warning(
        "Completion rate for second run: {:.2f}".format(np.array(check_2).mean())
    )

outcome = (
    df_test.join(
        pl.scan_parquet(
            data_dir
            / f"{args.tto_version}-tokenized"
            / "test"
            / "tokens_timelines_outcomes.parquet"
        ),
        how="left",
        on="hospitalization_id",
        validate="1:1",
    )
    .select("same_admission_death")
    .collect()
    .to_numpy()
    .ravel()
)

M0, M1, M2 = map(np.array, (M0, M1, M2))

for name, estm in {"M0": M0, "M1": M1, "M2": M2}.items():
    logger.info(f"{name=}".upper().ljust(42, "="))
    log_classification_metrics(y_true=outcome, y_score=estm, logger=logger)
    logger.info(bootstrap_ci(y_true=outcome, y_score=estm))


logger.info("---fin")
