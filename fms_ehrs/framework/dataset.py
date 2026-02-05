#!/usr/bin/env python3

"""
provide datasets for training

Supports two modes:
1. Standard mode (packed/padded): Loads input_ids only for language model training
2. Representation mode (padded only): Additionally loads numeric_values and relative_times
   for Experiment 2 soft discretization, xVal, and Time2Vec temporal encoding
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


def compute_relative_times_hours(times_list: list, *, t0: typing.Any | None = None) -> list[float]:
    """Convert a list of timestamps to relative hours since admission time.

    This respects MIMIC-IV's deidentification policy: "A single date shift was
    assigned to each subject_id. As a result, the data for a single patient are
    internally consistent." We use relative time (hours since admission) rather
    than absolute timestamps to avoid spurious cross-patient temporal signals.

    Important for windowed training:
    - When slicing an admission into multiple windows, we must keep the same
      reference admission start time across all windows. Otherwise, later windows
      would be (incorrectly) re-zeroed to start at 0 hours.

    Parameters
    ----------
    times_list : list
        List of datetime objects or None values
    t0 : optional
        Explicit admission start timestamp to use as the reference.
        If None, we infer t0 as the first non-null element of `times_list`
        (legacy behavior).

    Returns
    -------
    list[float]
        Relative time in hours since first non-null timestamp; None for null inputs
    """
    if not times_list or all(t is None for t in times_list):
        return [0.0] * len(times_list) if times_list else []

    # Find first non-null timestamp as admission time (unless provided).
    if t0 is None:
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


def _windowed_padded_examples(
    *,
    tokens: list[int],
    times: list | None,
    numeric_values: list | None,
    window_len: int,
    window_stride: int,
    pad_id: int,
    cont_id: int | None,
    max_windows: int | None = None,
) -> dict[str, list]:
    """
    Slice one admission timeline into overlapping fixed-length windows.

    Returns a dict-of-lists compatible with HF datasets map() with batched=True.

    Conventions:
    - Window 0 uses tokens[0:window_len] (no continuation marker).
    - Window k>0 prepends TL_CONT (if cont_id is provided) and then uses
      tokens[start : start + (window_len - 1)].
    - Times/numeric_values are sliced consistently. TL_CONT gets time=None and numeric=NaN.
    - Each window is padded to length window_len with PAD, time=None, numeric=NaN.
    - Relative times are computed w.r.t. the *admission-level* t0 (first non-null time in full timeline).

    Token-exposure knobs:
    - window_stride controls how we advance the window start index.
      * If cont_id is provided and window_stride == window_len, we interpret this as
        "non-overlapping contiguous coverage": window0 covers L tokens, and subsequent
        windows advance by (L-1) so that *raw tokens* are seen exactly once while
        leaving room for the TL_CONT marker.
      * Otherwise, we use regular range(0, n, window_stride) semantics (which may overlap).
    - max_windows (optional) caps windows per admission. If the natural number of
      windows exceeds max_windows, we deterministically subsample windows uniformly
      across the timeline while always including the first and last window.
    """
    if window_len < 2:
        raise ValueError(f"window_len must be >=2, got {window_len}")
    if window_stride < 1:
        raise ValueError(f"window_stride must be >=1, got {window_stride}")
    if max_windows is not None and int(max_windows) < 1:
        raise ValueError(f"max_windows must be >=1 when provided, got {max_windows}")

    n = len(tokens)
    if times is not None and len(times) != n:
        raise ValueError("times must align with tokens (same length)")
    if numeric_values is not None and len(numeric_values) != n:
        raise ValueError("numeric_values must align with tokens (same length)")

    # Admission-level reference time for Time2Vec (shared across windows).
    t0 = None
    if times:
        for ts in times:
            if ts is not None:
                t0 = ts
                break

    out_input_ids: list[list[int]] = []
    out_numeric: list[list[float]] = []
    out_rel_times: list[list[float]] = []

    def _nan():
        return float("nan")

    # Ensure we always emit at least one window, even for empty sequences.
    if n == 0:
        starts = [0]
    else:
        # Special case: when using TL_CONT windows, a stride of exactly window_len
        # would introduce 1-token "gaps" because subsequent windows only have room
        # for (window_len - 1) raw tokens. We treat window_stride == window_len as
        # an explicit request for *non-overlapping* contiguous coverage.
        if cont_id is not None and window_stride == window_len:
            starts = [0]
            if n > window_len:
                # Subsequent windows cover (window_len - 1) raw tokens each.
                # Start at window_len so token (window_len-1) is not repeated.
                starts.extend(list(range(window_len, n, window_len - 1)))
        else:
            starts = list(range(0, n, window_stride))

    # Cap windows per admission (deterministic subsampling).
    if max_windows is not None and len(starts) > int(max_windows):
        m = int(max_windows)
        if m == 1:
            starts = [starts[0]]
        else:
            last = len(starts) - 1
            # Uniform indices in [0, last], inclusive (strictly increasing since len(starts) > m).
            idxs = [(i * last) // (m - 1) for i in range(m)]
            starts = [starts[i] for i in idxs]

    for w, start in enumerate(starts):
        # Stop if the window would be entirely beyond the end.
        if n > 0 and start >= n:
            break

        use_cont = (w > 0) and (cont_id is not None)
        take = window_len - (1 if use_cont else 0)
        sl = tokens[start : start + take]

        win_tokens = ([cont_id] + sl) if use_cont else list(sl)
        # Pad tokens
        if len(win_tokens) < window_len:
            win_tokens = win_tokens + [pad_id] * (window_len - len(win_tokens))
        else:
            win_tokens = win_tokens[:window_len]

        out_input_ids.append(win_tokens)

        if numeric_values is not None:
            sln = numeric_values[start : start + take]
            win_num = ([_nan()] + [float(v) if v is not None else _nan() for v in sln]) if use_cont else [
                float(v) if v is not None else _nan() for v in sln
            ]
            if len(win_num) < window_len:
                win_num = win_num + [_nan()] * (window_len - len(win_num))
            else:
                win_num = win_num[:window_len]
            out_numeric.append(win_num)

        if times is not None:
            slt = times[start : start + take]
            win_times = ([None] + list(slt)) if use_cont else list(slt)
            if len(win_times) < window_len:
                win_times = win_times + [None] * (window_len - len(win_times))
            else:
                win_times = win_times[:window_len]
            out_rel_times.append(compute_relative_times_hours(win_times, t0=t0))

        # If the remaining tail is short, we still want to cover it. Stop when we've passed it.
        if n > 0 and (start + take) >= n:
            break

    out: dict[str, list] = {"input_ids": out_input_ids}
    if numeric_values is not None:
        out["numeric_values"] = out_numeric
    if times is not None:
        out["relative_times"] = out_rel_times
    return out


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
        If True, load padded_numeric_values for soft discretization and xVal.
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
        max_seq_length: int = int(os.getenv("IRB_MAX_SEQ_LENGTH", "4096")),
        shuffle_buffer_size: int = 1024,
        i_part: int = None,
        n_parts: int = None,
        include_numeric_values: bool = False,
        include_times: bool = False,
        # Windowed padded mode (Exp2/Exp3 full-timeline training):
        # When enabled, we *do not* rely on the tokenization-time `padded` column (which truncates).
        # Instead, we slice the full variable-length `tokens` column into overlapping windows.
        windowed_padded: bool = False,
        window_stride: int | None = None,
        add_cont_token: bool = False,
        # Extreme-length guardrail (compute stability):
        # Cap windows emitted per admission in windowed padded mode. If None, emit all windows.
        max_windows_per_admission: int | None = None,
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
        self.windowed_padded = windowed_padded
        self.window_stride = window_stride
        self.add_cont_token = add_cont_token
        self.max_windows_per_admission = max_windows_per_admission

        # Representation mode (numeric_values or times) requires padded collation
        if (include_numeric_values or include_times) and collation != "padded":
            raise ValueError(
                "include_numeric_values and include_times require collation='padded'. "
                "Packed collation destroys per-admission structure needed for relative times."
            )

        # Determine which columns to load
        #
        # NOTE:
        # - Standard padded mode uses the pre-materialized `padded` column (fixed length, may truncate).
        # - Windowed padded mode uses `tokens` (full timeline) and creates fixed-length windows here.
        if collation == "padded" and self.windowed_padded:
            base_columns = ["tokens"]
        else:
            base_columns = ["padded" if collation == "padded" else "tokens"]
        extra_columns = []
        if include_numeric_values:
            extra_columns.append("numeric_values" if (collation == "padded" and self.windowed_padded) else "padded_numeric_values")
        if include_times:
            extra_columns.append("times" if (collation == "padded" and self.windowed_padded) else "padded_times")

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
            # Windowed padded mode: explode each admission into multiple windows of length max_seq_length.
            if self.collation == "padded" and self.windowed_padded:
                pad_id = int(self.vocab("PAD"))
                cont_id = int(self.vocab("TL_CONT")) if self.add_cont_token and ("TL_CONT" in self.vocab.lookup) else None
                # Default stride: non-overlapping windows.
                #
                # With TL_CONT enabled, subsequent windows carry (window_len - 1) raw tokens
                # plus the marker. In _windowed_padded_examples, window_stride == window_len is
                # treated as an explicit request for non-overlapping contiguous coverage.
                stride = int(self.window_stride) if self.window_stride is not None else int(self.max_seq_length)
                max_w = int(self.max_windows_per_admission) if self.max_windows_per_admission is not None else None

                out_all: dict[str, list] = {"input_ids": []}
                if self.include_numeric_values:
                    out_all["numeric_values"] = []
                if self.include_times:
                    out_all["relative_times"] = []

                toks_list = batch["tokens"]
                times_list = batch.get("times")
                nums_list = batch.get("numeric_values")

                for i, toks in enumerate(toks_list):
                    times = times_list[i] if times_list is not None else None
                    nums = nums_list[i] if nums_list is not None else None
                    out_i = _windowed_padded_examples(
                        tokens=list(toks) if toks is not None else [],
                        times=list(times) if times is not None else None,
                        numeric_values=list(nums) if nums is not None else None,
                        window_len=int(self.max_seq_length),
                        window_stride=stride,
                        pad_id=pad_id,
                        cont_id=cont_id,
                        max_windows=max_w,
                    )
                    out_all["input_ids"].extend(out_i["input_ids"])
                    if self.include_numeric_values:
                        out_all["numeric_values"].extend(out_i.get("numeric_values", []))
                    if self.include_times:
                        out_all["relative_times"].extend(out_i.get("relative_times", []))

                return out_all

            # Standard (non-windowed) behavior.
            result = {"input_ids": batch["padded" if self.collation == "padded" else "tokens"]}

            if self.include_numeric_values and "padded_numeric_values" in batch:
                numeric_values = []
                for seq in batch["padded_numeric_values"]:
                    numeric_values.append([float(v) if v is not None else float("nan") for v in seq])
                result["numeric_values"] = numeric_values

            if self.include_times and "padded_times" in batch:
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
            schema_cols = {"hospitalization_id", "tokens", "times", "numeric_values", "seq_len", "padded"}

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
