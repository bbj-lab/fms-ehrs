#!/usr/bin/env python3

"""
learn the tokenizer on the training set and apply it to the validation and test sets
"""

import os
import pathlib

import fire as fi

from logger import get_logger
from tokenizer import ClifTokenizer

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    data_dir: os.PathLike = "../clif-data/",
    data_version: str = "day_stays_qc",
    vocab_path: os.PathLike = None,
    max_padded_len: int = 1024,
    day_stay_filter: bool = True,
    include_24h_cut: bool = True,
):
    data_dir = pathlib.Path(data_dir).expanduser().resolve()
    splits = ("train", "val", "test")

    for cut_at_24h in (False, True) if include_24h_cut else (False,):
        v = data_version + ("_first_24h" if cut_at_24h else "")

        dirs_in = dict()
        dirs_out = dict()
        for s in splits:
            dirs_in[s] = data_dir.joinpath("raw", s)
            dirs_out[s] = data_dir.joinpath(f"{v}-tokenized", s)
            dirs_out[s].mkdir(exist_ok=True, parents=True)

        # tokenize training set
        tkzr = ClifTokenizer(
            data_dir=dirs_in["train"],
            vocab_path=(
                pathlib.Path(vocab_path).expanduser().resolve()
                if vocab_path is not None
                else (
                    data_dir.joinpath(
                        f"{data_version}-tokenized", "train", "vocab.gzip"
                    )
                    if cut_at_24h
                    else None
                )
            ),
            max_padded_len=max_padded_len,
            day_stay_filter=day_stay_filter,
            cut_at_24h=cut_at_24h,
        )
        tokens_timelines = tkzr.get_tokens_timelines()
        tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
        tokens_timelines.write_parquet(
            dirs_out["train"].joinpath("tokens_timelines.parquet")
        )
        tkzr.vocab.save(dirs_out["train"].joinpath("vocab.gzip"))

        # take the learned tokenizer and tokenize the validation and test sets
        for s in ("val", "test"):
            tkzr = ClifTokenizer(
                data_dir=dirs_in[s],
                vocab_path=(
                    pathlib.Path(vocab_path).expanduser().resolve()
                    if vocab_path is not None
                    else data_dir.joinpath(
                        f"{data_version}-tokenized", "train", "vocab.gzip"
                    )
                ),
                max_padded_len=max_padded_len,
                day_stay_filter=day_stay_filter,
                cut_at_24h=cut_at_24h,
            )
            tokens_timelines = tkzr.get_tokens_timelines()
            tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
            tokens_timelines.write_parquet(
                dirs_out[s].joinpath("tokens_timelines.parquet")
            )


if __name__ == "__main__":
    fi.Fire(main)
