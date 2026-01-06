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
            "centiles", "deciles", "ventiles", "trentiles"
        ] = "deciles",
        clinical_anchoring: typing.Literal[
            "none", "5-10-5", "10-10-10"
        ] = "none",
        include_time_spacing_tokens: bool = False,
        fused_category_values: bool = False,
        detect_discrete: bool = False,
        include_ref_ranges: bool = False,
        **kwargs,
    ):
        """
        if no vocabulary is provided, we are in training mode; otherwise, the
        provided vocabulary is frozen

        Parameters
        ----------
        quantizer : str
            Binning strategy: "deciles" (10), "ventiles" (20), "trentiles" (30),
            or "centiles" (100)
        clinical_anchoring : str
            Reference range-aware binning allocation. Only applicable when
            include_ref_ranges=True. Options:
            - "none": No clinical anchoring (population-based quantiles)
            - "5-10-5": 5 bins below, 10 within, 5 above ref range (for ventiles)
            - "10-10-10": 10 bins in each region (for trentiles)
        include_ref_ranges : bool
            If True, use reference ranges for clinically-anchored binning
        """
        self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
        self.quantizer = quantizer
        self.clinical_anchoring = clinical_anchoring
        if quantizer == "centiles":
            self.q_tokens = tuple(map(lambda i: f"Q{i}", range(100)))
        elif quantizer == "deciles":
            self.q_tokens = tuple(map(lambda i: f"Q{i}", range(10)))
        elif quantizer == "ventiles":
            self.q_tokens = tuple(map(lambda i: f"Q{i}", range(20)))
        elif quantizer == "trentiles":
            self.q_tokens = tuple(map(lambda i: f"Q{i}", range(30)))
        self.special: tuple = ("TL_START", "TL_END", "PAD", "TRUNC", None, "nan")
        self.max_padded_length = max_padded_len
        self.include_time_spacing_tokens = include_time_spacing_tokens
        self.fused_category_values = fused_category_values
        self.detect_discrete = detect_discrete
        self.include_ref_ranges = include_ref_ranges
        # Validate clinical anchoring configuration
        if self.include_ref_ranges:
            if self.quantizer not in ("ventiles", "trentiles"):
                raise NotImplementedError(
                    "Ref ranges option only works with ventiles or trentiles quantizer."
                )
            if self.quantizer == "ventiles" and self.clinical_anchoring not in (
                "none", "5-10-5"
            ):
                raise ValueError(
                    f"Clinical anchoring '{clinical_anchoring}' not valid for ventiles. "
                    "Use 'none' or '5-10-5'."
                )
            if self.quantizer == "trentiles" and self.clinical_anchoring not in (
                "none", "10-10-10"
            ):
                raise ValueError(
                    f"Clinical anchoring '{clinical_anchoring}' not valid for trentiles. "
                    "Use 'none' or '10-10-10'."
            )
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
            year_seconds = 365.25 * 24 * 60 * 60
            month_seconds = year_seconds / 12
            self.t_breakpoints = (
                datetime.timedelta(minutes=5).total_seconds(),
                datetime.timedelta(minutes=15).total_seconds(),
                datetime.timedelta(hours=1).total_seconds(),
                datetime.timedelta(hours=2).total_seconds(),
                datetime.timedelta(hours=6).total_seconds(),
                datetime.timedelta(hours=12).total_seconds(),
                datetime.timedelta(days=1).total_seconds(),
                datetime.timedelta(days=3).total_seconds(),
                datetime.timedelta(days=7).total_seconds(),
                datetime.timedelta(days=14).total_seconds(),
                month_seconds,
                3 * month_seconds,
                6 * month_seconds,
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

    def set_quants(
        self,
        v: np.ndarray,
        c: str,
        prefix: str = None,
        ref_range_lower=None,
        ref_range_upper=None,
    ) -> None:
        """store training quantile information in the self.vocab object"""
        designator = f"{prefix}_{c}" if prefix is not None else c
        if not self.vocab.has_aux(designator) and self.vocab.is_training:
            if self.detect_discrete and len(unq := np.unique(v[np.isfinite(v)])) < len(
                self.q_tokens
            ):
                self.vocab.set_aux(designator, unq.tolist())
            elif self.quantizer == "ventiles":
                if (
                    self.include_ref_ranges
                    and ref_range_lower is not None
                    and ref_range_upper is not None
                    and self.clinical_anchoring != "none"
                ):
                    breaks = self._compute_anchored_breaks(
                        v, ref_range_lower, ref_range_upper, self.clinical_anchoring
                    )
                    self.vocab.set_aux(designator, np.unique(breaks))
                else:
                    self.vocab.set_aux(
                        designator,
                        np.nanquantile(v, np.arange(0.05, 1.0, 0.05)).tolist(),
                    )
            elif self.quantizer == "trentiles":
                if (
                    self.include_ref_ranges
                    and ref_range_lower is not None
                    and ref_range_upper is not None
                    and self.clinical_anchoring != "none"
                ):
                    breaks = self._compute_anchored_breaks(
                        v, ref_range_lower, ref_range_upper, self.clinical_anchoring
                    )
                    self.vocab.set_aux(designator, np.unique(breaks))
                else:
                    # Standard trentile: 30 bins = 29 breakpoints
                    self.vocab.set_aux(
                        designator,
                        np.nanquantile(v, np.arange(1/30, 1.0, 1/30)).tolist(),
                    )
            elif self.quantizer == "centiles":
                self.vocab.set_aux(
                    designator, np.nanquantile(v, np.arange(0.01, 1.0, 0.01)).tolist()
                )
            elif self.quantizer == "deciles":
                self.vocab.set_aux(
                    designator, np.nanquantile(v, np.arange(0.1, 1.0, 0.1)).tolist()
                )

    def _compute_anchored_breaks(
        self,
        v: np.ndarray,
        ref_range_lower: float,
        ref_range_upper: float,
        anchoring: str,
    ) -> list:
        """Compute clinically-anchored quantile breaks based on reference ranges.

        Parameters
        ----------
        v : np.ndarray
            Array of numeric values
        ref_range_lower : float
            Lower bound of reference range
        ref_range_upper : float
            Upper bound of reference range
        anchoring : str
            Anchoring strategy, e.g., "5-10-5" or "10-10-10"

        Returns
        -------
        list
            List of break points for np.digitize
        """
        # Parse anchoring pattern
        parts = anchoring.split("-")
        n_below = int(parts[0])
        n_within = int(parts[1])
        n_above = int(parts[2])

        # Partition values by reference range
        below = v[v < ref_range_lower]
        within = v[(ref_range_lower <= v) & (v <= ref_range_upper)]
        above = v[v > ref_range_upper]

        breaks = []

        # Below reference range: n_below bins = n_below-1 internal breaks + 1 boundary
        if len(below) > 0 and n_below > 0:
            below_quantiles = np.linspace(0, 1, n_below + 1)[1:-1]
            if len(below_quantiles) > 0:
                breaks.extend(np.nanquantile(below, below_quantiles).tolist())

        # Reference lower bound is a break point
        breaks.append(ref_range_lower)

        # Within reference range: n_within bins = n_within-1 internal breaks
        if len(within) > 0 and n_within > 1:
            within_quantiles = np.linspace(0, 1, n_within + 1)[1:-1]
            if len(within_quantiles) > 0:
                breaks.extend(np.nanquantile(within, within_quantiles).tolist())

        # Reference upper bound is a break point
        breaks.append(ref_range_upper)

        # Above reference range: n_above bins = n_above-1 internal breaks
        if len(above) > 0 and n_above > 0:
            above_quantiles = np.linspace(0, 1, n_above + 1)[1:-1]
            if len(above_quantiles) > 0:
                breaks.extend(np.nanquantile(above, above_quantiles).tolist())

        return sorted(breaks)

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
        if self.include_ref_ranges and {
            "ref_range_lower",
            "ref_range_upper",
        }.issubset(set(x.collect_schema().keys())):
            ref_range_lower = (
                y.row(0)[0]
                if len(y := x.select(pl.col("ref_range_lower").drop_nulls().mode())) > 0
                else None
            )
            ref_range_upper = (
                y.row(0)[0]
                if len(y := x.select(pl.col("ref_range_upper").drop_nulls().mode())) > 0
                else None
            )
            self.set_quants(
                v=v,
                c=c,
                prefix=prefix,
                ref_range_lower=ref_range_lower,
                ref_range_upper=ref_range_upper,
            )
        else:
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
                .with_columns(
                    # Align raw numeric values to token positions:
                    #   [code_token, quantile_token] -> [null, value]
                    numeric_values=pl.concat_list(
                        pl.lit(None).cast(pl.Float32), pl.col("value").cast(pl.Float32)
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
                .with_columns(
                    # Fused token represents a numeric event in a single position.
                    numeric_values=pl.concat_list(pl.col("value").cast(pl.Float32))
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
        self,
        tokens: np.ndarray,
        times: np.ndarray,
        numeric_values: np.ndarray | None = None,
    ) -> dict[str, list]:
        """insert tokens corresponding to the passage of time

        If numeric_values is provided, inserts null numeric values for the inserted
        time-spacing tokens so alignment with tokens/times is preserved.
        """
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
        if numeric_values is not None:
            assert len(numeric_values) == len(tokens)
            # Insert nulls for time-spacing tokens. We force object dtype so None is preserved
            # (NaN would incorrectly appear as a "present" numeric value).
            numeric_values_obj = np.array(numeric_values, dtype=object)
            insert_vals = np.array([None] * len(ix), dtype=object)
            new_numeric_values = np.insert(
                numeric_values_obj, np.array(ix) + 1, insert_vals
        )
        new_times = (
            np.insert(times, np.array(ix) + 1, np.array(times)[np.array(ix) + 1])
            .astype("datetime64[ms]")
            .astype(datetime.datetime)
        )  # spacing tokens assigned to the time at the end of the space
        ret = {"tokens": list(new_tokens), "times": list(new_times)}
        if numeric_values is not None:
            ret["numeric_values"] = list(new_numeric_values)
        return ret

    @staticmethod
    def cut_at_time(
        tokens_timelines: Frame, duration: pl.Duration = pl.duration(days=1)
    ) -> Frame:
        """allows us to select the first 24h of someone's timeline for predictive purposes"""
        schema_names = tokens_timelines.collect_schema().names()

        tt = tokens_timelines.with_columns(
                first_fail_or_0=(
                pl.col("times").list.eval(pl.element() - pl.col("").min() <= duration)
                ).list.arg_min()
        ).with_columns(
                valid_length=pl.when(pl.col("first_fail_or_0") == 0)
                .then(pl.col("times").list.len())
                .otherwise(pl.col("first_fail_or_0"))
        ).with_columns(
            pl.col("times").list.head(pl.col("valid_length")).alias("times"),
            pl.col("tokens").list.head(pl.col("valid_length")).alias("tokens"),
            )

        if "numeric_values" in schema_names:
            tt = tt.with_columns(
                pl.col("numeric_values")
                .list.head(pl.col("valid_length"))
                .alias("numeric_values")
            )

        return tt.filter(pl.col("times").list.max() - pl.col("times").list.min() <= duration)

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
            # Optional aligned padding for times / numeric values (used by Exp2 time2vec/encoders)
            if "times" in tokens_timelines.collect_schema().names():
                tt_under = tt_under.with_columns(
                    padded_times=pl.concat_list(
                        "times",
                        pl.lit(None)
                        .cast(pl.Datetime(time_unit="ms"))
                        .repeat_by(self.max_padded_length - pl.col("seq_len")),
                    )
                )
            if "numeric_values" in tokens_timelines.collect_schema().names():
                tt_under = tt_under.with_columns(
                    padded_numeric_values=pl.concat_list(
                        "numeric_values",
                        pl.lit(None)
                        .cast(pl.Float32)
                        .repeat_by(self.max_padded_length - pl.col("seq_len")),
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
            if "times" in tokens_timelines.collect_schema().names():
                tt_over = tt_over.with_columns(
                    padded_times=pl.concat_list(
                        pl.col("times").list.slice(
                            offset=0, length=self.max_padded_length - 1
                        ),
                        pl.lit(None).cast(pl.Datetime(time_unit="ms")),
                    )
                )
            if "numeric_values" in tokens_timelines.collect_schema().names():
                tt_over = tt_over.with_columns(
                    padded_numeric_values=pl.concat_list(
                        pl.col("numeric_values").list.slice(
                            offset=0, length=self.max_padded_length - 1
                        ),
                        pl.lit(None).cast(pl.Float32),
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
                .map_elements(tokenizer.get_token_type_from_int, return_dtype=pl.String)
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
