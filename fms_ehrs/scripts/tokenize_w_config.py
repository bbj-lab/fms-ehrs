#!/usr/bin/env python3

"""
learn the tokenizer on the training set and apply it to the validation and test sets;
training folders will contain a copy of the vocabulary object & configuration
"""

import argparse
import pathlib
import shutil

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import fix_perms, set_perms
from fms_ehrs.framework.tokenizer import Tokenizer21
from fms_ehrs.framework.tokenizer_base import summarize

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../tmp-test/")
parser.add_argument("--data_version_in", type=str, default="raw")
parser.add_argument("--data_version_out", type=str, default="test")
parser.add_argument("--vocab_path", type=pathlib.Path, default=None)
parser.add_argument("--include_24h_cut", action="store_true")
parser.add_argument(
    "--config_loc", type=pathlib.Path, default="../fms_ehrs/config/clif-21.yaml"
)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

# make output sub-directories
data_dir = pathlib.Path(args.data_dir).expanduser().resolve()
splits = ("train", "val", "test")

dirs_in = dict()
dirs_out = dict()
for s in splits:
    dirs_in[s] = data_dir.joinpath(args.data_version_in, s)
    dirs_out[s] = data_dir.joinpath(args.data_version_out + "-tokenized", s)
    dirs_out[s].mkdir(exist_ok=True, parents=True)
    fix_perms(data_dir.joinpath(args.data_version_out + "-tokenized"))
    fix_perms(dirs_out[s])

if args.include_24h_cut:
    dirs_out_24h = dict()
    for s in splits:
        dirs_out_24h[s] = data_dir.joinpath(
            args.data_version_out + "_first_24h-tokenized", s
        )
        dirs_out_24h[s].mkdir(exist_ok=True, parents=True)
        fix_perms(data_dir.joinpath(args.data_version_out + "_first_24h-tokenized"))
        fix_perms(dirs_out_24h[s])

# tokenize training set
tkzr = Tokenizer21(
    data_dir=dirs_in["train"],
    vocab_path=(
        pathlib.Path(args.vocab_path).expanduser().resolve()
        if args.vocab_path is not None
        else None
    ),
    cut_at_24h=False,
    config_file=args.config_loc,
)
tokens_timelines = tkzr.get_tokens_timelines()
logger.info("train...")
summarize(tkzr, tokens_timelines, logger=logger)
tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
set_perms(tokens_timelines.write_parquet)(
    dirs_out["train"].joinpath("tokens_timelines.parquet")
)
tkzr.vocab.save(dirs_out["train"].joinpath("vocab.gzip"))
shutil.copy2(
    pathlib.Path(args.config_loc).expanduser().resolve(),
    dirs_out["train"] / "config.yaml",
)

if args.include_24h_cut:
    tokens_timelines_24h = tkzr.cut_at_time(tokens_timelines)
    logger.info("24h cut...")
    summarize(tkzr, tokens_timelines_24h, logger=logger)
    tokens_timelines_24h = tkzr.pad_and_truncate(tokens_timelines_24h)
    set_perms(tokens_timelines.write_parquet)(
        dirs_out_24h["train"].joinpath("tokens_timelines.parquet")
    )
    tkzr.vocab.save(dirs_out_24h["train"].joinpath("vocab.gzip"))
    shutil.copy2(
        pathlib.Path(args.config_loc).expanduser().resolve(),
        dirs_out_24h["train"] / "config.yaml",
    )

# take the learned tokenizer and tokenize the validation and test sets
for s in ("val", "test"):
    tkzr = Tokenizer21(
        data_dir=dirs_in[s],
        vocab_path=(
            pathlib.Path(args.vocab_path).expanduser().resolve()
            if args.vocab_path is not None
            else data_dir.joinpath(
                f"{args.data_version_out}-tokenized", "train", "vocab.gzip"
            )
        ),
        cut_at_24h=False,
        config_file=args.config_loc,
    )
    tokens_timelines = tkzr.get_tokens_timelines()
    logger.info(f"{s}...")
    summarize(tkzr, tokens_timelines, logger=logger)
    tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
    set_perms(tokens_timelines.write_parquet)(
        dirs_out[s].joinpath("tokens_timelines.parquet")
    )
    if args.include_24h_cut:
        tokens_timelines_24h = tkzr.cut_at_time(tokens_timelines)
        logger.info("24h cut...")
        summarize(tkzr, tokens_timelines_24h, logger=logger)
        tokens_timelines_24h = tkzr.pad_and_truncate(tokens_timelines_24h)
        set_perms(tokens_timelines_24h.write_parquet)(
            dirs_out_24h[s].joinpath("tokens_timelines.parquet")
        )

logger.info("---fin")
