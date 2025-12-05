#!/usr/bin/env python3

"""
a framework on which tokenizers may be built
"""

import datetime
import logging
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
        quantizer: typing.Literal[
            "centiles", "deciles", "sigmas", "ventiles"
        ] = "deciles",
        include_time_spacing_tokens: bool = False,
        fused_category_values: bool = False,
    ):
        """
        if no vocabulary is provided, we are in training mode; otherwise, the
        provided vocabulary is frozen
        """
        self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
        self.quantizer = quantizer
        if quantizer == "centiles":
            self.q_tokens = tuple(map(lambda i: f"Q{i}", range(100)))
        elif quantizer == "deciles":
            self.q_tokens = tuple(map(lambda i: f"Q{i}", range(10)))
        elif quantizer == "ventiles":
            self.q_tokens = tuple(map(lambda i: f"Q{i}", range(20)))
        elif quantizer == "sigmas":
            self.q_tokens = ("Q3-", "Q2-", "Q1-", "Q0-", "Q0+", "Q1+", "Q2+", "Q3+")
        self.special: tuple = ("TL_START", "TL_END", "PAD", "TRUNC", None, "nan")
        self.max_padded_length = max_padded_len
        self.include_time_spacing_tokens = include_time_spacing_tokens
        self.fused_category_values = fused_category_values
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
                (self.q_tokens if not fused_category_values else tuple())
                + self.special
                + (self.t_tokens if self.include_time_spacing_tokens else tuple())
            )
            self.vocab.is_training = True
        else:
            self.vocab_path = pathlib.Path(vocab_path).expanduser().resolve()
            self.vocab = Vocabulary().load(self.vocab_path)
            self.vocab.is_training = False

    def set_quants(self, v: np.ndarray, c: str, prefix: str = None) -> None:
        """store training quantile information in the self.vocab object"""
        designator = f"{prefix}_{c}" if prefix is not None else c
        if not self.vocab.has_aux(designator) and self.vocab.is_training:
            if self.quantizer == "centiles":
                self.vocab.set_aux(
                    designator, np.nanquantile(v, np.arange(0.01, 1.0, 0.01)).tolist()
                )
            elif self.quantizer == "deciles":
                self.vocab.set_aux(
                    designator, np.nanquantile(v, np.arange(0.1, 1.0, 0.1)).tolist()
                )
            elif self.quantizer == "ventiles":
                self.vocab.set_aux(
                    designator, np.nanquantile(v, np.arange(0.05, 1.0, 0.05)).tolist()
                )
            elif self.quantizer == "sigmas":
                μ = np.nanmean(v)
                σ = np.nanstd(v) + np.finfo(float).eps
                self.vocab.set_aux(designator, (μ + σ * np.arange(-3, 4)).tolist())

    def get_quants(self, v: np.ndarray, c: str, prefix: str = None) -> pl.Expr:
        """obtain corresponding quantiles using self.vocab object"""
        designator = f"{prefix}_{c}" if prefix is not None else c
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

    def process_single_category(self, x: Frame, prefix: str) -> Frame:
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
        self.set_quants(v=v, c=c, prefix=prefix)
        if not self.fused_category_values:
            return (
                x.with_columns(
                    token=pl.lit(self.vocab(f"{prefix}_{c}")).cast(pl.Int64),
                    token_quantile=self.get_quants(v=v, c=c, prefix=prefix),
                )
                .with_columns(
                    tokens=pl.concat_list("token", "token_quantile").cast(
                        pl.List(pl.Int64)
                    )
                )
                .drop("token", "token_quantile")
            )
        else:
            return (
                x.with_columns(quantile=self.get_quants(v=v, c=c, prefix=prefix))
                .with_columns(
                    tokens=pl.concat_list(
                        pl.col("quantile").map_elements(
                            lambda x, c=c, prefix=prefix: self.vocab(
                                f"{prefix}_{c}_Q{x}"
                            ),
                            return_dtype=pl.Int64,
                            skip_nulls=False,
                        )
                    )
                )
                .drop("quantile")
            )

    def process_cat_val_frame(self, df: Frame, label: str) -> Frame:
        """handle tables that can mostly be described in terms of categories and
        values"""
        return pl.concat(
            self.process_single_category(x, label) for x in df.partition_by("category")
        )

    def time_spacing_inserter(
        self, tokens: np.ndarray, times: np.ndarray
    ) -> dict[str, list]:
        """insert tokens corresponding to the passage of time"""
        assert len(tokens) == len(times)
        binned = np.digitize(
            np.diff(times).astype("timedelta64[s]").astype(int), bins=self.t_breakpoints
        )
        ix = np.flatnonzero(binned)
        td = binned[ix]
        new_tokens = np.insert(
            tokens,
            np.array(ix) + 1,  # insert *after* ix
            np.array(list(map(self.vocab, self.t_tokens)))[
                np.array(td) - 1
            ],  # when td is 0, it lies before our first breakpoint (<5min)
        )
        new_times = (
            np.insert(times, np.array(ix) + 1, np.array(times)[np.array(ix) + 1])
            .astype("datetime64[ms]")
            .astype(datetime.datetime)
        )  # spacing tokens assigned to the time at the end of the space
        return {"tokens": list(new_tokens), "times": list(new_times)}

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

    def get_token_type(self, tk: str | None) -> str:
        """determine the type of a token, usually specified in the token's prefix"""
        if tk in self.special:
            return "SPECIAL"
        elif tk in self.q_tokens:
            return "QUANT"
        elif self.include_time_spacing_tokens and tk in self.t_tokens:
            return "T_SPACER"
        else:
            return tk.split("_")[0]

    def get_token_type_from_int(self, ti: int) -> str:
        return self.get_token_type(self.vocab.reverse[ti])


def summarize(
    tokenizer: BaseTokenizer,
    tokens_timelines: Frame,
    k: int = 20,
    logger: logging.Logger = None,
) -> None:
    """provide posthoc summary statistics"""

    post = logger.info if logger is not None else print

    post("Timelines generated: {}".format(tokens_timelines.shape[0]))
    post("Vocabulary size: {}".format(len(tokenizer.vocab)))

    post(
        "Summary stats of timeline lengths: \n {}".format(
            tokens_timelines.select(pl.col("tokens").list.len()).describe()
        )
    )

    for s in range(3):
        post(
            "Example timeline: \n {}".format(
                [
                    tokenizer.vocab.reverse[t]
                    for t in tokens_timelines.sample(1, seed=s).select("tokens").item()
                ]
            )
        )

    post(
        "Summary stats of timeline duration: \n {}".format(
            tokens_timelines.select(
                pl.col("times").list.min().alias("start_time"),
                pl.col("times").list.max().alias("end_time"),
            )
            .select((pl.col("end_time") - pl.col("start_time")).alias("duration"))
            .describe()
        )
    )

    with pl.Config(tbl_rows=len(tokenizer.vocab)):
        post(
            "Top {k} tokens by usage: \n {out}".format(
                k=k,
                out=tokens_timelines.select("tokens")
                .explode("tokens")
                .rename({"tokens": "token"})
                .join(tokenizer.vocab.get_frame(), on="token")
                .select("word")
                .to_series()
                .value_counts()
                .sort("count", descending=True)
                .head(k),
            )
        )

    with pl.Config(tbl_rows=len(tokenizer.vocab)):
        post(
            tokens_timelines.select(
                pl.col("tokens")
                .explode()
                .map_elements(tokenizer.get_token_type_from_int, return_dtype=str)
            )
            .to_series()
            .value_counts()
            .sort(by="count", descending=True)
        )


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
    tt = eg_tkzr.time_spacing_inserter(eg_tokens, eg_times)
    print(list(map(eg_tkzr.vocab.reverse.__getitem__, tt["tokens"])))
    # ['Q0', 'Q1', 'T_5m-15m', 'Q2', 'Q3', 'Q4', 'T_1h-2h', 'Q5', 'Q6']
    print(tt["times"])
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

    print(eg_tkzr.get_token_type("Q3"))
    # QUANT
    print(eg_tkzr.get_token_type(None))
    # SPECIAL
    print(eg_tkzr.get_token_type_from_int(3))
    # QUANT
