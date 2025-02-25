#!/usr/bin/env python3

"""
provide datasets for training, validation, and inference
"""

import itertools
import os
import pathlib
import typing

import numpy as np
import polars as pl
import torch as t
import datasets as ds

from vocabulary import Vocabulary

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class Datasets:

    def __init__(
        self,
        data_version: str,
        hm: Pathlike,
        collation: typing.Literal["padded", "packed"] = "packed",
        *,
        max_seq_length: int = 1024,
        shuffle_buffer_size: int = 1024,
    ):
        self.data_version = data_version
        self.hm = pathlib.Path(hm)
        self.collation = collation
        self.max_seq_length = max_seq_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.t_rng = t.Generator().manual_seed(42)
        self.n_rng = np.random.default_rng(42)
        self.splits = ("train", "val", "test")
        self.data_dirs = {
            s: self.hm.joinpath("clif-data", f"{self.data_version}-tokenized", s)
            for s in self.splits
        }
        self.vocab = Vocabulary().load(self.data_dirs["train"].joinpath("vocab.gzip"))
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
                        "padded" if self.collation == "padded" else "tokens"
                    ]
                },
                batched=True,
                remove_columns=[
                    "hospitalization_id",
                    "tokens",
                    "times",
                    "seq_len",
                ],
                features=ds.Features({"input_ids": ds.Sequence(ds.Value("uint8"))}),
            )
            .with_format("torch")
        )
        self.n_train: int = self.dataset["train"].num_rows
        self.n_val: int = self.dataset["val"].num_rows
        self.n_test: int = self.dataset["test"].num_rows

    def generate_padding(self, poisson_rate: float = 7.0):
        tk: int = self.vocab("PAD")
        size = t.poisson(t.tensor(poisson_rate), generator=self.t_rng).to(t.uint8)
        return t.full(size=(size.item(),), fill_value=tk, dtype=t.uint8)

    def chunk_iterable(self, it):
        ret: t.Tensor = t.Tensor(size=(0,))
        for eg in it:
            x = t.concat((eg["input_ids"], self.generate_padding()))
            while x.size(dim=0) > 0:
                ndiff = min(self.max_seq_length - ret.size(dim=0), x.size(dim=0))
                ret = t.concat((ret, x[:ndiff]))
                x = x[ndiff:]
                if ret.size(dim=0) == self.max_seq_length:
                    yield {"input_ids": ret.to(t.uint8)}
                    ret = t.Tensor(size=(0,))

    def rt_padding_to_left(self, t_rt):
        tk: int = self.vocab("PAD")
        i = t.argmax((t_rt == tk).int()).item()
        return t.concat([t.full((t_rt.shape[0] - i,), tk), t_rt[:i]])

    def get_train_dataset(self, n_epochs: int = 10):
        if self.collation == "padded":
            return self.dataset["train"].shuffle(generator=self.n_rng)
        elif self.collation == "packed":
            return ds.IterableDataset.from_generator(
                lambda: self.chunk_iterable(
                    ds.IterableDataset.from_generator(
                        lambda: itertools.chain.from_iterable(
                            itertools.repeat(iter(self.dataset["train"]), n_epochs)
                        )
                    ).shuffle(
                        generator=self.n_rng, buffer_size=self.shuffle_buffer_size
                    )
                ),
                features=ds.Features({"input_ids": ds.Sequence(ds.Value("uint8"))}),
            )
        else:
            raise ValueError(
                "collation should be `padded` or `packed`, not ", self.collation
            )

    def get_val_dataset(self):
        if self.collation == "padded":
            return self.dataset["val"]
        elif self.collation == "packed":
            return ds.IterableDataset.from_generator(
                lambda: self.chunk_iterable(self.dataset["val"]),
                features=ds.Features({"input_ids": ds.Sequence(ds.Value("uint8"))}),
            )
        else:
            raise ValueError(
                "collation should be `padded` or `packed`, not ", self.collation
            )

    def get_test_dataset(self):
        return self.dataset["test"]

    def get_context_length(self):
        return self.dataset["test"].select(range(1))["input_ids"].shape[1]

    def get_test_set_for_predictions(self):
        return self.dataset["test"].map(
            lambda x: {"input_ids": self.rt_padding_to_left(x["padded"])},
            remove_columns=["padded"],
        )
