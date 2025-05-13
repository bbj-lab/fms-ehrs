#!/usr/bin/env python3

"""
learn the tokenizer on the training set and apply it to the validation and test sets
"""

import os
import pathlib
import typing

import fire as fi

from src.framework.logger import get_logger
from src.framework.tokenizer import ClifTokenizer, summarize

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    data_dir: os.PathLike = None,
    data_version_in: str = "raw",
    data_version_out: str = "day_stays",
    vocab_path: os.PathLike = None,
    max_padded_len: int = 1024,
    day_stay_filter: bool = True,
    include_24h_cut: bool = True,
    valid_admission_window: tuple[str, str] = None,
    lab_time: typing.Literal["collect", "result"] = "result",
    quantizer: typing.Literal["deciles", "sigmas"] = "deciles",
    drop_deciles: bool = False,
    drop_nulls_nans: bool = False,
):
    data_dir = pathlib.Path(data_dir).expanduser().resolve()
    splits = ("train", "val", "test")

    for cut_at_24h in (False, True) if include_24h_cut else (False,):
        logger.info(f"{cut_at_24h=}...")
        v = data_version_out + ("_first_24h" if cut_at_24h else "")

        dirs_in = dict()
        dirs_out = dict()
        for s in splits:
            dirs_in[s] = data_dir.joinpath(data_version_in, s)
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
                        f"{data_version_out}-tokenized", "train", "vocab.gzip"
                    )
                    if cut_at_24h
                    else None
                )
            ),
            max_padded_len=max_padded_len,
            day_stay_filter=day_stay_filter,
            cut_at_24h=cut_at_24h,
            valid_admission_window=valid_admission_window,
            quantizer=quantizer,
            lab_time=lab_time,
            drop_deciles=drop_deciles,
            drop_nulls_nans=drop_nulls_nans,
        )
        tokens_timelines = tkzr.get_tokens_timelines()
        logger.info("train...")
        summarize(tkzr, tokens_timelines, logger=logger)
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
                        f"{data_version_out}-tokenized", "train", "vocab.gzip"
                    )
                ),
                max_padded_len=max_padded_len,
                day_stay_filter=day_stay_filter,
                cut_at_24h=cut_at_24h,
                valid_admission_window=valid_admission_window,
                quantizer=quantizer,
                lab_time=lab_time,
                drop_deciles=drop_deciles,
                drop_nulls_nans=drop_nulls_nans,
            )
            tokens_timelines = tkzr.get_tokens_timelines()
            logger.info(f"{s}...")
            summarize(tkzr, tokens_timelines, logger=logger)
            tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
            tokens_timelines.write_parquet(
                dirs_out[s].joinpath("tokens_timelines.parquet")
            )


if __name__ == "__main__":
    fi.Fire(main)
