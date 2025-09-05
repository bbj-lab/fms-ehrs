#!/usr/bin/env python3

"""
a framework on which tokenizers may be built
"""

import os
import pathlib
import typing

import numpy as np
import pandas as pd
import polars as pl

from fms_ehrs.framework.vocabulary import Vocabulary

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class BaseTokenizer:
    def __init__(
        self,
        data_dir: Pathlike = pathlib.Path("../.."),
        vocab_path: Pathlike = None,
        max_padded_len: int = None,
        quantizer: typing.Literal["deciles", "sigmas"] = "deciles",
        include_time_spacing_tokens: bool = False,
    ):
        """
        if no vocabulary is provided, we are in training mode; otherwise, the
        provided vocabulary is frozen
        """
        self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
        self.quantizer = quantizer
        self.q_tokens = (
            tuple(map(lambda i: f"Q{i}", range(10)))
            if self.quantizer == "deciles"
            else ("Q3-", "Q2-", "Q1-", "Q0-", "Q0+", "Q1+", "Q2+", "Q3+")
        )
        self.special: tuple = ("TL_START", "TL_END", "PAD", "TRUNC", None, "nan")
        self.max_padded_length = max_padded_len
        self.include_time_spacing_tokens = include_time_spacing_tokens
        if self.include_time_spacing_tokens:
            self.t_tokens = (
                "T_5m-15m",
                "T_15m-1h",
                "T_1h-2h",
                "T_2h-6h",
                "T_6h-12h",
                "T_12h-1d",
                "T_1d-3d",
                "T_3d-1w",
                "T_1w-2w",
                "T_2w-1mt",
                "T_1mt-3mt",
                "T_3mt-6mt",
                "T_6mt+",
            )
            self.t_breakpoints = (
                pd.Timedelta("5 minutes").total_seconds(),
                pd.Timedelta("15 minutes").total_seconds(),
                pd.Timedelta("1 hour").total_seconds(),
                pd.Timedelta("2 hours").total_seconds(),
                pd.Timedelta("6 hours").total_seconds(),
                pd.Timedelta("12 hours").total_seconds(),
                pd.Timedelta("1 day").total_seconds(),
                pd.Timedelta("3 days").total_seconds(),
                pd.Timedelta("7 days").total_seconds(),
                pd.Timedelta("14 days").total_seconds(),
                pd.Timedelta("365.25 days").total_seconds() / 12,
                3 * pd.Timedelta("365.25 days").total_seconds() / 12,
                6 * pd.Timedelta("365.25 days").total_seconds() / 12,
            )
        if vocab_path is None:
            self.vocab_path = None
            self.vocab = Vocabulary(
                self.q_tokens
                + self.special
                + (self.t_tokens if self.include_time_spacing_tokens else tuple())
            )
            self.vocab.is_training = True
        else:
            self.vocab_path = pathlib.Path(vocab_path).expanduser().resolve()
            self.vocab = Vocabulary().load(self.vocab_path)
            self.vocab.is_training = False

    def set_quants(self, v: np.array, c: str, label: str = None) -> None:
        """store training quantile information in the self.vocab object"""
        designator = f"{label}_{c}" if label is not None else c
        if not self.vocab.has_aux(designator) and self.vocab.is_training:
            if self.quantizer == "deciles":
                self.vocab.set_aux(
                    designator, np.nanquantile(v, np.arange(0.1, 1.0, 0.1)).tolist()
                )
            elif self.quantizer == "sigmas":
                μ = np.nanmean(v)
                σ = np.nanstd(v) + np.finfo(float).eps
                self.vocab.set_aux(designator, (μ + σ * np.arange(-3, 4)).tolist())

    def get_quants(self, v: np.array, c: str, label: str = None) -> pl.Expr:
        """obtain corresponding quantiles using self.vocab object"""
        designator = f"{label}_{c}" if label is not None else c
        return pl.lit(
            (
                pl.Series(
                    np.where(
                        np.isfinite(v),
                        np.digitize(v, bins=self.vocab.get_aux(designator)),
                        self.vocab("nan"),
                    )
                )
                if self.vocab.has_aux(designator)
                else self.vocab(None)
            )
        ).cast(pl.Int64)

    def process_single_category(self, x: Frame, label: str) -> Frame:
        """
        Quantize a sub-table consisting of a single category

        The way our quantization works, if a category takes on only a single
        value, then this value is sent to the Q9 token, because, e.g.
        `np.digitize(1, bins=[1] * 9) == 9`
        and:
        `np.digitize(
        [1, 2],
        bins=np.nanquantile([1, 1, 1, 2, 2, 2, 2], np.arange(0.1, 1.0, 0.1)),
        ) == [3, 9]`
        This is why the Q9 token appears quite a bit more often in our dataset than
        certain other quantile tokens.
        """
        v = x.select("value").to_numpy().ravel()
        c = x.select("category").row(0)[0]
        self.set_quants(v=v, c=c, label=label)
        return (
            x.with_columns(
                token=pl.lit(self.vocab(f"{label}_{c}")).cast(pl.Int64),
                token_quantile=self.get_quants(v=v, c=c, label=label),
            )
            .filter(~pl.col("token").is_in([self.vocab(None), self.vocab("nan")]))
            .filter(
                ~pl.col("token_quantile").is_in([self.vocab(None), self.vocab("nan")])
            )
            .with_columns(
                tokens=pl.concat_list("token", "token_quantile").cast(
                    pl.List(pl.Int64)
                ),
                times=pl.concat_list("event_time", "event_time").cast(
                    pl.List(pl.Datetime(time_unit="ms"))
                ),
            )
        )

    def process_cat_val_frame(self, df: Frame, label: str) -> Frame:
        """handle tables that can mostly be described in terms of categories and
        values"""
        return pl.concat(
            self.process_single_category(x, label) for x in df.partition_by("category")
        )

    def time_spacing_inserter(self, tokens, times):
        assert len(tokens) == len(times)
        tdiffs = np.diff(times).astype("timedelta64[s]").astype(int)
        try:
            ix, td = zip(
                *filter(
                    lambda x: x[1] > 0,
                    enumerate(np.digitize(tdiffs, bins=self.t_breakpoints).tolist()),
                )
            )
        except ValueError:  # no digitized tdiffs > 0; no insertions to be made
            assert np.count_nonzero(np.digitize(tdiffs, bins=self.t_breakpoints)) == 0
            return {"tokens": tokens, "times": times}
        new_tokens = np.insert(
            tokens,
            np.array(ix) + 1,  # insert *after* ix
            np.array(list(map(self.vocab, self.t_tokens)))[
                np.array(td) - 1
            ],  # when td is 0, it lies before our first breakpoint (<5min)
        )
        new_times = np.insert(
            times, np.array(ix) + 1, np.array(times)[np.array(ix) + 1]
        ).astype(
            np.datetime64
        )  # spacing tokens assigned to the time at the end of the space
        return {"tokens": new_tokens, "times": new_times}

    @staticmethod
    def cut_at_time(
        tokens_timelines: Frame, duration: pl.Duration = pl.duration(days=1)
    ) -> Frame:
        """allows us to select the first 24h of someone's timeline for predictive purposes"""
        tt = (
            tokens_timelines.with_columns(
                first_fail_or_0=(
                    pl.col("times").list.eval(
                        pl.element() - pl.col("").min() <= duration
                    )
                ).list.arg_min()
            )
            .with_columns(
                valid_length=pl.when(pl.col("first_fail_or_0") == 0)
                .then(pl.col("times").list.len())
                .otherwise(pl.col("first_fail_or_0"))
            )
            .with_columns(
                pl.col("times").list.head(pl.col("valid_length")),
                pl.col("tokens").list.head(pl.col("valid_length")),
            )
            .filter(pl.col("times").list.max() - pl.col("times").list.min() <= duration)
        )
        return tt

    def pad_and_truncate(self, tokens_timelines: Frame) -> Frame:
        if self.max_padded_length is not None:
            tt = tokens_timelines.lazy().with_columns(
                seq_len=pl.col("tokens").list.len()
            )
            tt_under = tt.filter(
                pl.col("seq_len") <= self.max_padded_length
            ).with_columns(
                padded=pl.concat_list(
                    "tokens",
                    pl.lit(self.vocab("PAD")).repeat_by(
                        self.max_padded_length - pl.col("seq_len")
                    ),
                )
            )
            tt_over = tt.filter(
                pl.col("seq_len") > self.max_padded_length
            ).with_columns(
                padded=pl.concat_list(
                    pl.col("tokens").list.slice(
                        offset=0, length=self.max_padded_length - 1
                    ),
                    pl.lit(self.vocab("TRUNC")),
                )
            )
            return pl.concat([tt_under, tt_over]).collect()
        else:
            return tokens_timelines

    def print_aux(self) -> None:
        self.vocab.print_aux()


if __name__ == "__main__":
    # exhibit time spacing inserter logic
    eg_tokens = np.arange(7)
    eg_times = np.array(
        [
            "2000-01-01T00:00:00",
            "2000-01-01T00:00:00",  # 7 min space here
            "2000-01-01T00:07:00",
            "2000-01-01T00:07:00",
            "2000-01-01T00:07:00",  # 1 hr 23 min space here
            "2000-01-01T01:30:00",
            "2000-01-01T01:30:00",
        ],
        dtype="datetime64[s]",
    )
    eg_tkzr = BaseTokenizer(include_time_spacing_tokens=True)
    new_tt = eg_tkzr.time_spacing_inserter(eg_tokens, eg_times)
    print(list(map(eg_tkzr.vocab.reverse.__getitem__, new_tt["tokens"])))
    # ['Q0', 'Q1', 'T_5m-15m', 'Q2', 'Q3', 'Q4', 'T_1h-2h', 'Q5', 'Q6']
    print(new_tt["times"].tolist())
    # [
    #   datetime.datetime(2000, 1, 1, 0, 0),
    #   datetime.datetime(2000, 1, 1, 0, 0),
    #   datetime.datetime(2000, 1, 1, 0, 7), # spacer is assigned end time
    #   datetime.datetime(2000, 1, 1, 0, 7),
    #   datetime.datetime(2000, 1, 1, 0, 7),
    #   datetime.datetime(2000, 1, 1, 0, 7),
    #   datetime.datetime(2000, 1, 1, 1, 30), # spacer is assigned end time
    #   datetime.datetime(2000, 1, 1, 1, 30),
    #   datetime.datetime(2000, 1, 1, 1, 30)
    # ]
