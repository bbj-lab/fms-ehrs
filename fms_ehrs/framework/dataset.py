#!/usr/bin/env python3

"""
provide datasets for training

Supports two modes:
1. Standard mode (packed/padded): Loads input_ids only for language model training
2. Representation mode (padded only): Additionally loads numeric_values and relative_times
   for Experiment 2 soft/continuous encoders and Time2Vec temporal encoding
"""

import itertools
import os
import pathlib
import typing
from datetime import datetime

import datasets as ds
import numpy as np
import polars as pl
import torch as t

from fms_ehrs.framework.vocabulary import Vocabulary

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


def compute_relative_times_hours(times_list: list) -> list[float]:
    """Convert a list of timestamps to relative hours since first timestamp.

    This respects MIMIC-IV's deidentification policy: "A single date shift was
    assigned to each subject_id. As a result, the data for a single patient are
    internally consistent." We use relative time (hours since admission) rather
    than absolute timestamps to avoid spurious cross-patient temporal signals.

    Parameters
    ----------
    times_list : list
        List of datetime objects or None values

    Returns
    -------
    list[float]
        Relative time in hours since first non-null timestamp; None for null inputs
    """
    if not times_list or all(t is None for t in times_list):
        return [0.0] * len(times_list) if times_list else []

    # Find first non-null timestamp as admission time
    t0 = None
    for ts in times_list:
        if ts is not None:
            t0 = ts
            break

    if t0 is None:
        return [0.0] * len(times_list)

    result = []
    for ts in times_list:
        if ts is None:
            result.append(0.0)  # Padding positions get 0
        else:
            # Handle both datetime objects and numeric timestamps
            if isinstance(ts, (int, float)):
                # Assume milliseconds since epoch
                delta_ms = ts - (t0 if isinstance(t0, (int, float)) else t0.timestamp() * 1000)
                result.append(delta_ms / (1000 * 3600))  # Convert ms to hours
            elif isinstance(ts, datetime):
                delta = ts - t0
                result.append(delta.total_seconds() / 3600)
            else:
                result.append(0.0)
    return result


class Datasets:
    """Dataset provider for EHR model training.

    Parameters
    ----------
    data_version : str
        Name of the data version (e.g., "day_stays")
    data_dir : Pathlike
        Root directory containing tokenized data
    collation : {"padded", "packed"}
        Collation strategy. "packed" concatenates multiple sequences for efficiency.
        "padded" keeps one sequence per row (required for representation mode).
    max_seq_length : int
        Maximum sequence length for packed collation
    shuffle_buffer_size : int
        Buffer size for shuffling iterable datasets
    i_part, n_parts : int, optional
        For distributed training: partition index and total partitions
    include_numeric_values : bool
        If True, load padded_numeric_values for soft/continuous encoders.
        Requires padded collation.
    include_times : bool
        If True, load padded_times and compute relative_times for Time2Vec.
        Requires padded collation.
    """

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
        include_numeric_values: bool = False,
        include_times: bool = False,
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
            s: self.data_dir / f"{self.data_version}-tokenized" / s for s in self.splits
        }
        self.vocab = Vocabulary().load(self.data_dirs["train"] / "vocab.gzip")
        self.uint_dtype = (
            t.uint8 if len(self.vocab) <= t.iinfo(t.uint8).max else t.int64
        )
        self.i_part = i_part
        self.n_parts = n_parts
        self.include_numeric_values = include_numeric_values
        self.include_times = include_times

        # Representation mode (numeric_values or times) requires padded collation
        if (include_numeric_values or include_times) and collation != "padded":
            raise ValueError(
                "include_numeric_values and include_times require collation='padded'. "
                "Packed collation destroys per-admission structure needed for relative times."
            )

        # Determine which columns to load
        base_columns = ["padded" if collation == "padded" else "tokens"]
        extra_columns = []
        if include_numeric_values:
            extra_columns.append("padded_numeric_values")
        if include_times:
            extra_columns.append("padded_times")

        # Build the dataset
        self.dataset = self._build_dataset(base_columns, extra_columns)

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

    def _build_dataset(self, base_columns: list, extra_columns: list):
        """Build the HuggingFace dataset with appropriate columns."""
        all_columns = base_columns + extra_columns

        # Load raw dataset
        raw_dataset = ds.load_dataset(
            "parquet",
            data_files={
                s: str(self.data_dirs[s] / "tokens_timelines.parquet")
                for s in self.splits
            },
        )

        # Define the mapping function
        def process_batch(batch):
            result = {
                "input_ids": batch[
                    "padded" if self.collation == "padded" else "tokens"
                ]
            }

            if self.include_numeric_values and "padded_numeric_values" in batch:
                # Convert to float tensor, replacing None with NaN
                numeric_values = []
                for seq in batch["padded_numeric_values"]:
                    numeric_values.append([
                        float(v) if v is not None else float("nan")
                        for v in seq
                    ])
                result["numeric_values"] = numeric_values

            if self.include_times and "padded_times" in batch:
                # Compute relative times in hours
                relative_times = []
                for seq in batch["padded_times"]:
                    relative_times.append(compute_relative_times_hours(seq))
                result["relative_times"] = relative_times

            return result

        # Determine columns to remove (everything not in our output)
        # We need to inspect actual schema
        sample_file = self.data_dirs["train"] / "tokens_timelines.parquet"
        if sample_file.exists():
            schema_cols = set(pl.scan_parquet(sample_file).collect_schema().names())
        else:
            schema_cols = {"hospitalization_id", "tokens", "times", "seq_len", "padded"}

        # Define output features
        features_dict = {
            "input_ids": ds.Sequence(
                ds.Value(str(self.uint_dtype).split(".")[-1])
            )
        }
        if self.include_numeric_values:
            features_dict["numeric_values"] = ds.Sequence(ds.Value("float32"))
        if self.include_times:
            features_dict["relative_times"] = ds.Sequence(ds.Value("float32"))

        # Columns to remove (all original columns that aren't in our output)
        remove_cols = list(schema_cols - set(features_dict.keys()))

        return (
            raw_dataset
            .map(
                process_batch,
                batched=True,
                remove_columns=remove_cols,
                features=ds.Features(features_dict),
            )
            .with_format("torch")
        )

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

    def get_train_dataset(self, n_epochs: int = 10, iterable: bool = False):
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

    def get_val_dataset(self, iterable: bool = False):
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
