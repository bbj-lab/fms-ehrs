#!/usr/bin/env python3

"""
learn the tokenizer on the training set and apply it to the validation and test sets
"""

import pathlib

import fire as fi

from tokenizer import ClifTokenizer
from logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    data_hm=pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-data"),
    data_version: str = "day_stays_qc",
    max_padded_len: int = 1024,
    day_stay_filter: bool = True,
):
    data_hm = pathlib.Path(data_hm).expanduser().resolve()
    splits = ("train", "val", "test")

    for cut_at_24h in (False, True):
        v = data_version + ("_first_24h" if cut_at_24h else "")

        data_dirs = dict()
        out_dirs = dict()
        for s in splits:
            data_dirs[s] = data_hm.joinpath("raw", s)
            out_dirs[s] = data_hm.joinpath(f"{v}-tokenized", s)
            out_dirs[s].mkdir(exist_ok=True, parents=True)

        # tokenize training set
        tkzr = ClifTokenizer(
            data_dir=data_dirs["train"],
            vocab_path=(
                data_hm.joinpath(f"{data_version}-tokenized", "train", "vocab.gzip")
                if cut_at_24h
                else None
            ),
            max_padded_len=max_padded_len,
            day_stay_filter=day_stay_filter,
            cut_at_24h=cut_at_24h,
        )
        tokens_timelines = tkzr.get_tokens_timelines()
        tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
        tokens_timelines.write_parquet(
            out_dirs["train"].joinpath("tokens_timelines.parquet")
        )
        tkzr.vocab.save(out_dirs["train"].joinpath("vocab.gzip"))

        # take the learned tokenizer and tokenize the validation and test sets
        for s in ("val", "test"):
            tkzr = ClifTokenizer(
                data_dir=data_dirs[s],
                vocab_path=(
                    data_hm.joinpath(f"{data_version}-tokenized", "train", "vocab.gzip")
                ),
                max_padded_len=max_padded_len,
                day_stay_filter=day_stay_filter,
                cut_at_24h=cut_at_24h,
            )
            tokens_timelines = tkzr.get_tokens_timelines()
            tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
            tokens_timelines.write_parquet(
                out_dirs[s].joinpath("tokens_timelines.parquet")
            )


if __name__ == "__main__":
    fi.Fire(main)
