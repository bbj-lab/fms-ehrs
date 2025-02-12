#!/usr/bin/env python3

"""
provides a generic class for creating and maintaining a map from a vocabulary
of strings to unique integers
"""

import collections
import functools
import gzip
import pathlib
import pickle
import typing
import warnings

import polars as pl

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Hashable: typing.TypeAlias = collections.abc.Hashable
Pathlike: typing.TypeAlias = pathlib.PurePath | str


class Vocabulary:
    """
    maintains a dictionary `lookup` mapping words -> tokens,
    a dictionary `reverse` inverting the lookup, and a dictionary
    `aux` mapping words -> auxiliary info
    """

    def __init__(self, words: tuple = (), *, is_training: bool = True):
        assert len(set(words)) == len(words)
        self.lookup = {v: i for i, v in enumerate(words)}
        self.reverse = dict(enumerate(words))
        self.aux = {}
        self._is_training = is_training

    def __call__(self, word: Hashable) -> int | None:
        try:
            return self.lookup[word]
        except KeyError:
            if self._is_training:
                self.lookup[word], self.reverse[n] = (n := len(self.lookup)), word
                return n
            else:
                warnings.warn(
                    "Encountered previously unseen token: {} {}".format(
                        word, type(word)
                    )
                )
                return None

    def set_aux(self, word: Hashable, aux_data):
        if self._is_training:
            self.aux[word] = aux_data
        else:
            raise Exception("Tokenizer is frozen after training.")
        return self

    def has_aux(self, word: Hashable) -> bool:
        return word in self.aux

    def in_lookup(self, word: Hashable) -> bool:
        return word in self.lookup

    def get_aux(self, word: Hashable):
        return self.aux[word]

    def save(self, filepath: Pathlike) -> typing.Self:
        with gzip.open(pathlib.Path(filepath).expanduser(), "w+") as f:
            pickle.dump(
                {
                    "lookup": self.lookup,
                    "reverse": self.reverse,
                    "aux": {k: list(v) for k, v in self.aux.items()},
                },
                f,
            )
        return self

    def load(self, filepath: Pathlike) -> typing.Self:
        with gzip.open(pathlib.Path(filepath).expanduser(), mode="r+") as f:
            for k, v in pickle.load(f).items():
                setattr(self, k, v)
        return self

    def get_frame(self) -> Frame:
        return pl.from_records(
            list(self.lookup.items()), schema=("word", "token"), orient="row"
        )

    def __len__(self) -> int:
        return len(self.lookup)

    @property
    def is_training(self) -> bool:
        return self._is_training

    @is_training.setter
    def is_training(self, value: bool):
        self._is_training = value

    def print_aux(self):
        for k, v in self.aux.items():
            print(
                "{k}: {v}".format(
                    k=k, v=list(map(functools.partial(round, ndigits=2), v))
                )
            )


if __name__ == "__main__":

    import tempfile

    import numpy as np

    rng = np.random.default_rng(42)

    v1 = Vocabulary(tuple(map(lambda i: f"Q{i}", range(10))))
    v1(42)
    v1.set_aux(42, np.sort(rng.integers(low=0, high=1000, size=9)))
    v1.print_aux()

    v2 = Vocabulary()

    with tempfile.NamedTemporaryFile() as fp:
        v1.save(fp.name)
        v2.load(fp.name)

    assert v1(42) == v2(42)
    assert v1.lookup == v2.lookup
    assert np.array_equal(v1.get_aux(42), v2.get_aux(42))

    print(v2.lookup)
    print(v2.reverse)
    print(v2.aux)
