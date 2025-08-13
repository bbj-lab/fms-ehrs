#!/usr/bin/env python3

"""
provide datasets for training
"""

import itertools
import os
import pathlib
import typing

import datasets as ds
import numpy as np
import polars as pl
import torch as t

from fms_ehrs.framework.vocabulary import Vocabulary

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class Datasets:
    def __init__(
        self,
        data_version: str,
        data_dir: Pathlike,
        collation: typing.Literal["padded", "packed"] = "packed",
        *,
        max_seq_length: int = 1024,
        shuffle_buffer_size: int = 1024,
        i_part: int = None,
        n_parts: int = None,
    ):
        self.data_version = data_version
        self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
        self.collation = collation
        self.max_seq_length = max_seq_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.t_rng = t.Generator().manual_seed(42)
        self.np_rng = np.random.default_rng(42)
        self.splits = ("train", "val")
        self.data_dirs = {
            s: self.data_dir.joinpath(f"{self.data_version}-tokenized", s)
            for s in self.splits
        }
        self.vocab = Vocabulary().load(self.data_dirs["train"].joinpath("vocab.gzip"))
        self.uint_dtype = (
            t.uint8 if len(self.vocab) <= t.iinfo(t.uint8).max else t.int64
        )
        self.i_part = i_part
        self.n_parts = n_parts
        self.dataset = (
            ds.load_dataset(
                "parquet",
                data_files={
                    s: str(self.data_dirs[s].joinpath("tokens_timelines.parquet"))
                    for s in self.splits
                },
            )
            .map(
                lambda batch: {
                    "input_ids": batch[
                        "padded" if (self.collation == "padded") else "tokens"
                    ]
                },
                batched=True,
                remove_columns=[
                    "hospitalization_id",
                    "tokens",
                    "times",
                    "seq_len",
                    "padded",
                ],
                features=ds.Features(
                    {
                        "input_ids": ds.Sequence(
                            ds.Value(str(self.uint_dtype).split(".")[-1])
                        )
                    }
                ),
            )
            .with_format("torch")
        )
        if self.i_part is not None or self.n_parts is not None:
            assert 0 <= self.i_part < self.n_parts
            for s in self.splits:
                self.dataset[s] = self.dataset[s].select(
                    np.array_split(np.arange(self.dataset[s].num_rows), self.n_parts)[
                        self.i_part
                    ]
                )
        self.n_train: int = self.dataset["train"].num_rows
        self.n_val: int = self.dataset["val"].num_rows

    def generate_padding(self, poisson_rate: float = 7.0):
        size = t.poisson(t.tensor(poisson_rate), generator=self.t_rng).to(
            self.uint_dtype
        )
        return t.full(
            size=(size.item(),), fill_value=self.vocab("PAD"), dtype=self.uint_dtype
        )

    def chunk_iterable(self, it):
        ret: t.Tensor = t.Tensor(size=(0,))
        for eg in it:
            x = t.concat((eg["input_ids"], self.generate_padding()))
            while x.size(dim=0) > 0:
                ndiff = min(self.max_seq_length - ret.size(dim=0), x.size(dim=0))
                ret = t.concat((ret, x[:ndiff]))
                x = x[ndiff:]
                if ret.size(dim=0) == self.max_seq_length:
                    yield {"input_ids": ret.to(self.uint_dtype)}
                    ret = t.Tensor(size=(0,))

    def get_train_dataset(self, n_epochs: int = 10, iterable: bool = True):
        if self.collation == "padded":
            x = self.dataset["train"].shuffle(generator=self.np_rng)
        elif self.collation == "packed":
            x = ds.IterableDataset.from_generator(
                lambda: self.chunk_iterable(
                    ds.IterableDataset.from_generator(
                        lambda: itertools.chain.from_iterable(
                            itertools.repeat(iter(self.dataset["train"]), n_epochs)
                        )
                    ).shuffle(
                        generator=self.np_rng, buffer_size=self.shuffle_buffer_size
                    )
                ),
                features=ds.Features(
                    {
                        "input_ids": ds.Sequence(
                            ds.Value(str(self.uint_dtype).split(".")[-1])
                        )
                    }
                ),
            )
        else:
            raise ValueError(
                "collation should be `padded` or `packed`, not ", self.collation
            )
        return x if iterable else ds.Dataset.from_list(list(x))

    def get_val_dataset(self, iterable: bool = True):
        if self.collation == "padded":
            x = self.dataset["val"]
        elif self.collation == "packed":
            x = ds.IterableDataset.from_generator(
                lambda: self.chunk_iterable(self.dataset["val"]),
                features=ds.Features(
                    {
                        "input_ids": ds.Sequence(
                            ds.Value(str(self.uint_dtype).split(".")[-1])
                        )
                    }
                ),
            )
        else:
            raise ValueError(
                "collation should be `padded` or `packed`, not ", self.collation
            )
        return x if iterable else ds.Dataset.from_list(list(x))

    def get_context_length(self):
        return len(self.dataset["train"].select(range(1))["input_ids"][0])


if __name__ == "__main__":
    if os.uname().nodename.startswith("cri"):
        hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")
    else:
        # change following line to develop locally
        hm = pathlib.Path("~/Documents/chicago/CLIF/")

    data_dir = hm.parent.joinpath(hm.stem + "-tokenized").expanduser()
    data = Datasets(
        data_version="clif-development-sample",
        data_dir=hm,
        collation="packed",
        i_part=42,
        n_parts=100,
    )
    tr = data.get_train_dataset(n_epochs=1, iterable=False)
