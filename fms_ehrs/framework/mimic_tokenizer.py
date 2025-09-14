#!/usr/bin/env python3

"""
provides a tokenizing interface to take raw MIMIC-IV data and convert
it to tokenized timelines at the hospitalization_id level
"""

import collections
import functools
import logging
import os
import pathlib
import re
import typing

import numpy as np
import pandas as pd
import polars as pl

from fms_ehrs.framework.vocabulary import Vocabulary
from fms_ehrs.framework.data_loader import MimicDataLoader

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class MimicTokenizer:
    """
    tokenizes a directory containing MIMIC-IV CSV files; works directly with
    raw MIMIC-IV data instead of CLIF-converted data
    """

    def __init__(
        self,
        *,
        data_dir: Pathlike = pathlib.Path("../.."),
        vocab_path: Pathlike = None,
        max_padded_len: int = None,
        day_stay_filter: bool = False,
        cut_at_24h: bool = False,
        valid_admission_window: tuple[str, str] = None,
        lab_time: typing.Literal["charttime", "storetime"] = "charttime",
        quantizer: typing.Literal["deciles", "sigmas"] = "deciles",
        drop_deciles: bool = False,
        drop_nulls_nans: bool = False,
        n_top_reports: int = 100,
        include_time_spacing_tokens: bool = False,
    ):
        """
        if no vocabulary is provided, we are in training mode; otherwise, the
        provided vocabulary is frozen
        """
        self.data_dir = pathlib.Path(data_dir).expanduser()
        self.tbl = dict()
        self.quantizer = quantizer
        self.q_tokens = (
            tuple(map(lambda i: f"Q{i}", range(10)))
            if self.quantizer == "deciles"
            else ("Q3-", "Q2-", "Q1-", "Q0-", "Q0+", "Q1+", "Q2+", "Q3+")
        )
        self.special = ("TL_START", "TL_END", "PAD", "TRUNC", None, "nan")
        if vocab_path is None:
            self.vocab_path = None
            self.vocab = Vocabulary(self.q_tokens + self.special)
            self.vocab.is_training = True
        else:
            self.vocab_path = pathlib.Path(vocab_path).expanduser()
            self.vocab = Vocabulary().load(self.vocab_path)
            self.vocab.is_training = False
        self.max_padded_length = max_padded_len
        self.day_stay_filter = bool(day_stay_filter)
        self.cut_at_24h = bool(cut_at_24h)
        self.valid_admission_window = valid_admission_window
        self.lab_time = lab_time
        self.drop_deciles = bool(drop_deciles)
        self.drop_nulls_nans = bool(drop_nulls_nans)
        self.n_top_reports = n_top_reports
        self.include_time_spacing_tokens = bool(include_time_spacing_tokens)
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

    def load_tables(self) -> None:
        """lazy-load all CSV tables from the MIMIC-IV directory structure"""
        # Use the MimicDataLoader to load tables
        data_loader = MimicDataLoader(self.data_dir)
        data_loader.load_tables()
        
        # Copy the loaded tables to self.tbl for compatibility
        self.tbl = data_loader.tables
    
    def _load_tables(self) -> None:
        """Load tables and update self.tbl (internal method for pipeline)"""
        data_loader = MimicDataLoader(self.data_dir)
        data_loader.load_tables()
        self.tbl = data_loader.tables
        
        # Apply data type fixes for consistent joins
        if "labevents" in self.tbl:
            self.tbl["labevents"] = self.tbl["labevents"].with_columns(
                pl.col("itemid").cast(pl.Utf8),
                pl.col("hadm_id").cast(pl.Utf8)
            )
        if "d_labitems" in self.tbl:
            self.tbl["d_labitems"] = self.tbl["d_labitems"].with_columns(
                pl.col("itemid").cast(pl.Utf8)
            )
        if "chartevents" in self.tbl:
            self.tbl["chartevents"] = self.tbl["chartevents"].with_columns(
                pl.col("itemid").cast(pl.Utf8),
                pl.col("hadm_id").cast(pl.Utf8)
            )
        if "d_items" in self.tbl:
            self.tbl["d_items"] = self.tbl["d_items"].with_columns(
                pl.col("itemid").cast(pl.Utf8)
            )
        if "admissions" in self.tbl:
            self.tbl["admissions"] = self.tbl["admissions"].with_columns(
                pl.col("hadm_id").cast(pl.Utf8)
            )
        if "icustays" in self.tbl:
            self.tbl["icustays"] = self.tbl["icustays"].with_columns(
                pl.col("hadm_id").cast(pl.Utf8)
            )
        
        # Fix invalid dates in sample data (2180 -> 2018) and convert to datetime
        if "labevents" in self.tbl:
            self.tbl["labevents"] = self.tbl["labevents"].with_columns(
                pl.col("charttime")
                .str.replace("2180", "2018")
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
            )
        if "chartevents" in self.tbl:
            self.tbl["chartevents"] = self.tbl["chartevents"].with_columns(
                pl.col("charttime")
                .str.replace("2180", "2018")
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
            )

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
                times=pl.concat_list("event_time", "event_time"),
            )
        )

    def process_cat_val_frame(self, df: Frame, label: str) -> Frame:
        """handle tables that can mostly be described in terms of categories and values"""
        return pl.concat(
            self.process_single_category(x, label) for x in df.partition_by("category")
        )

    def process_tables(self) -> None:
        """Process MIMIC-IV tables into tokenized format (legacy method)"""
        self.tbl = self._process_tables(self.tbl)
    
    def _process_tables(self, tables: dict) -> dict:
        """Pure function: Process MIMIC-IV tables into tokenized format"""
        processed_tables = tables.copy()
        
        # Process patients table
        processed_tables["patients"] = (
            tables["patients"]
            .group_by("subject_id")
            .agg(
                pl.col("gender").str.to_lowercase().first(),
                pl.col("anchor_age").first(),
            )
            .with_columns(
                pl.col("gender").map_elements(
                    lambda x: self.vocab(f"SEX_{x}"),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
            )
            .with_columns(
                tokens=pl.concat_list("gender")
            )
            .select("subject_id", "tokens", "anchor_age")
        )

        # Process admissions table
        processed_tables["admissions"] = (
            tables["admissions"]
            .group_by("hadm_id")
            .agg(
                pl.col("subject_id").first(),
                pl.col("admittime")
                .first()
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_start"),
                pl.col("dischtime")
                .first()
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_end"),
                pl.col("admission_type")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .first(),
                pl.col("hospital_expire_flag").first(),
            )
            .filter(
                pl.col("event_start").is_between(
                    pl.lit(self.valid_admission_window[0]).cast(pl.Date),
                    pl.lit(self.valid_admission_window[1]).cast(pl.Date),
                )
                if self.valid_admission_window is not None
                else True
            )
            .with_columns(
                pl.col("admission_type").map_elements(
                    lambda x: self.vocab(f"ADMN_{x}"),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
                pl.col("hospital_expire_flag").map_elements(
                    lambda x: self.vocab(f"DSCG_{'expired' if x == 1 else 'alive'}"),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
            )
            .select(
                "subject_id",
                "hadm_id",
                "event_start",
                "event_end",
                "admission_type",
                "hospital_expire_flag",
            )
            .sort(by="hadm_id")
        )

        # Join patients and admissions to get age at admission
        processed_tables["admissions"] = (
            processed_tables["admissions"]
            .join(processed_tables["patients"], on="subject_id", how="left")
            .with_columns(
                age_at_admission=pl.col("anchor_age")
            )
            .select(
                "subject_id", "hadm_id", "event_start", "event_end", 
                "age_at_admission", "admission_type", "hospital_expire_flag"
            )
        )

        # Process lab events
        processed_tables["labevents"] = (
            tables["labevents"]
            .join(tables["d_labitems"], on="itemid", how="left")
            .filter(~pl.col("category").is_null())
            .filter(~pl.col("hadm_id").is_null())
            .with_columns(
                # Convert valuenum from string to float, handling null values
                pl.col("valuenum").cast(pl.Float64, strict=False).alias("value")
            )
            .filter(~pl.col("value").is_null())  # Remove rows where value couldn't be converted
            .select(
                "hadm_id",
                pl.col(self.lab_time)
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_time"),
                pl.col("category").str.to_lowercase().alias("category"),
                "value",
            )
        )

        # Process chart events (vitals and other measurements)
        processed_tables["chartevents"] = (
            tables["chartevents"]
            .join(tables["d_items"], on="itemid", how="left")
            .filter(~pl.col("category").is_null())
            .filter(~pl.col("hadm_id").is_null())
            .with_columns(
                # Convert valuenum from string to float, handling null values
                pl.col("valuenum").cast(pl.Float64, strict=False).alias("value")
            )
            .filter(~pl.col("value").is_null())  # Remove rows where value couldn't be converted
            .select(
                "hadm_id",
                pl.col("charttime")
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_time"),
                pl.col("category").str.to_lowercase().alias("category"),
                "value",
            )
        )

        # Process ICU stays for transfer events
        processed_tables["icustays"] = (
            tables["icustays"]
            .with_columns(
                event_time=pl.col("intime").cast(pl.Datetime(time_unit="ms")),
                category=pl.lit("icu_admission"),
            )
            .with_columns(
                tokens=pl.col("category")
                .str.to_lowercase()
                .map_elements(
                    lambda x: [self.vocab(f"ADT_{x}")],
                    return_dtype=pl.List(pl.Int64),
                    skip_nulls=False,
                ),
                times=pl.col("event_time").map_elements(
                    lambda x: [x], return_dtype=pl.List(pl.Datetime), skip_nulls=False
                ),
            )
            .select("hadm_id", "event_time", "tokens", "times")
            .cast({"times": pl.List(pl.Datetime(time_unit="ms"))})
        )
        
        return processed_tables

    def _apply_quantization(self, tables: dict) -> dict:
        """Pure function: Apply quantization to tables"""
        quantized_tables = tables.copy()
        
        # Process lab and chart events with quantiles (need to collect for quantile computation)
        if "labevents" in quantized_tables:
            quantized_tables["labevents"] = self.process_cat_val_frame(
                quantized_tables["labevents"].collect(), label="LAB"
            )
        
        if "chartevents" in quantized_tables:
            quantized_tables["chartevents"] = self.process_cat_val_frame(
                quantized_tables["chartevents"].collect(), label="VTL"
            )

        # Process admissions with age quantiles (need to collect for quantile computation)
        admissions_collected = quantized_tables["admissions"].collect()
        c = "age_at_admission"
        v = admissions_collected.select(c).to_numpy().ravel()
        self.set_quants(v=v, c=c)
        quantized_tables["admissions"] = (
            admissions_collected
            .with_columns(age_at_admission=self.get_quants(v=v, c=c))
            .with_columns(admission_tokens=pl.concat_list(c, "admission_type"))
            .drop(c, "admission_type")
        )
        
        return quantized_tables

    def _combine_timelines(self, tables: dict) -> Frame:
        """Pure function: Combine all timelines into a single frame"""
        # Get admission frame
        admission_tokens = self._get_admission_frame(tables)
        
        # Get events frame
        events = self._get_events_frame(tables)
        
        # Get discharge frame
        discharge_tokens = self._get_discharge_frame(tables)
        
        # Combine the admission tokens, event tokens, and discharge tokens
        tt = (
            admission_tokens
            .lazy()
            .join(
                events,
                on="hadm_id",
                how="left",
                validate="1:1",
            )
            .join(
                discharge_tokens.lazy(),
                on="hadm_id",
                validate="1:1",
            )
            .with_columns(
                tokens=pl.concat_list("adm_tokens", "tokens", "dis_tokens"),
                times=pl.concat_list("adm_times", "times", "dis_times"),
            )
            .select("hadm_id", "tokens", "times")
            .sort(by="hadm_id")
        )
        
        return tt

    def _apply_filters(self, tt: Frame) -> Frame:
        """Pure function: Apply final filters to timelines"""
        if self.day_stay_filter:
            tt = tt.filter(
                (pl.col("times").list.get(-1) - pl.col("times").list.get(0))
                >= pl.duration(days=1)
            )

        if self.cut_at_24h:
            tt = self.cut_at_time(tt)

        if self.drop_deciles or self.drop_nulls_nans:
            filtered = (
                tt.explode("tokens", "times")
                .filter(pl.col("tokens") >= 10 if self.drop_deciles else pl.lit(True))
                .filter(
                    (~pl.col("tokens").is_in([self.vocab(None), self.vocab("nan")]))
                    if self.drop_nulls_nans
                    else pl.lit(True)
                )
            )
            new_times = filtered.group_by(
                "hadm_id", maintain_order=True
            ).agg([pl.col("times")])
            new_tokens = filtered.group_by(
                "hadm_id", maintain_order=True
            ).agg([pl.col("tokens")])
            tt = new_tokens.join(
                new_times,
                on="hadm_id",
                validate="1:1",
                maintain_order="left",
            )

        return tt.collect()

    def _get_admission_frame(self, tables: dict) -> Frame:
        """Pure function: Get admission frame from tables"""
        admission_tokens = (
            tables["patients"]
            .join(tables["admissions"], on="subject_id", validate="1:m")
            .cast({"event_start": pl.Datetime(time_unit="ms")})
            .with_columns(
                adm_tokens=pl.concat_list(
                    pl.lit(self.vocab("TL_START")),
                    pl.col("tokens"),
                    pl.col("admission_tokens"),
                ),
                adm_times=pl.concat_list(*[pl.col("event_start")] * 6),
            )
            .select(
                "hadm_id",
                pl.col("event_start").alias("event_time"),
                "adm_tokens",
                "adm_times",
            )
        )
        return admission_tokens

    def _get_events_frame(self, tables: dict) -> Frame:
        """Pure function: Get events frame from tables"""
        events = pl.concat(
            tables[k].select("hadm_id", "event_time", "tokens", "times")
            for k in tables.keys()
            if k not in ("patients", "admissions", "d_labitems", "d_items")
        )

        # Aggregate tokens and times by hadm_id
        tokens_agg = (
            events.lazy()
            .sort("event_time", pl.col("tokens").list.first())
            .group_by("hadm_id", maintain_order=True)
            .agg([pl.col("tokens").explode()])
        )

        times_agg = (
            events.lazy()
            .sort("event_time")
            .group_by("hadm_id", maintain_order=True)
            .agg([pl.col("times").explode()])
        )

        event_tokens = tokens_agg.join(
            times_agg, on="hadm_id", validate="1:1", maintain_order="left"
        )

        if self.include_time_spacing_tokens:
            event_tokens = event_tokens.with_columns(
                pl.struct(["tokens", "times"])
                .map_elements(
                    lambda x: self.time_spacing_inserter(x["tokens"], x["times"])[
                        "tokens"
                    ],
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("tokens"),
                pl.struct(["tokens", "times"])
                .map_elements(
                    lambda x: self.time_spacing_inserter(x["tokens"], x["times"])[
                        "times"
                    ],
                    return_dtype=pl.List(pl.Datetime(time_unit="ms")),
                )
                .alias("times"),
            )
        return event_tokens

    def _get_discharge_frame(self, tables: dict) -> Frame:
        """Pure function: Get discharge frame from tables"""
        discharge_tokens = (
            tables["admissions"]
            .rename({"event_end": "event_time"})
            .cast({"event_time": pl.Datetime(time_unit="ms")})
            .with_columns(
                dis_tokens=pl.concat_list(
                    "hospital_expire_flag", pl.lit(self.vocab("TL_END"))
                ),
                dis_times=pl.concat_list(*[pl.col("event_time")] * 2),
            )
            .cast({"dis_times": pl.List(pl.Datetime(time_unit="ms"))})
            .select("hadm_id", "event_time", "dis_tokens", "dis_times")
        )
        return discharge_tokens

    def run_times_qc(self) -> None:
        """Quality control for timeline boundaries"""
        alt_times = (
            self.tbl["chartevents"]
            .group_by("hadm_id")
            .agg(
                event_start_alt=pl.col("event_time").min(),
                event_end_alt=pl.col("event_time").max(),
            )
        )

        self.tbl["admissions"] = (
            self.tbl["admissions"]
            .join(alt_times, how="left", on="hadm_id", validate="1:1")
            .with_columns(
                event_start=pl.min_horizontal("event_start", "event_start_alt"),
                event_end=pl.max_horizontal("event_end", "event_end_alt"),
            )
            .drop("event_start_alt", "event_end_alt")
            .filter(pl.col("event_start") < pl.col("event_end"))
        )

    def get_admission_frame(self) -> Frame:
        """prepend patient-level tokens to each admission event"""
        admission_tokens = (
            self.tbl["patients"]
            .join(self.tbl["admissions"], on="subject_id", validate="1:m")
            .cast({"event_start": pl.Datetime(time_unit="ms")})
            .with_columns(
                adm_tokens=pl.concat_list(
                    pl.lit(self.vocab("TL_START")),
                    pl.col("tokens"),
                    pl.col("admission_tokens"),
                ),
                adm_times=pl.concat_list(*[pl.col("event_start")] * 6),
            )
            .select(
                "hadm_id",
                pl.col("event_start").alias("event_time"),
                "adm_tokens",
                "adm_times",
            )
        )

        return admission_tokens

    def get_discharge_frame(self) -> Frame:
        """gather discharge tokens"""
        discharge_tokens = (
            self.tbl["admissions"]
            .rename({"event_end": "event_time"})
            .cast({"event_time": pl.Datetime(time_unit="ms")})
            .with_columns(
                dis_tokens=pl.concat_list(
                    "hospital_expire_flag", pl.lit(self.vocab("TL_END"))
                ),
                dis_times=pl.concat_list(*[pl.col("event_time")] * 2),
            )
            .cast({"dis_times": pl.List(pl.Datetime(time_unit="ms"))})
            .select("hadm_id", "event_time", "dis_tokens", "dis_times")
        )

        return discharge_tokens

    def time_spacing_inserter(self, tokens, times):
        """Insert time spacing tokens between events"""
        assert len(tokens) == len(times)
        tdiffs = np.diff(times).astype("timedelta64[s]").astype(int)
        try:
            ix, td = zip(
                *filter(
                    lambda x: x[1] > 0,
                    enumerate(np.digitize(tdiffs, bins=self.t_breakpoints).tolist()),
                )
            )
        except ValueError:  # no digitized tdiffs > 0?
            assert np.count_nonzero(np.digitize(tdiffs, bins=self.t_breakpoints)) == 0
            return {"tokens": tokens, "times": times}
        new_tokens = np.insert(
            tokens,
            np.array(ix) + 1,  # insert *after* ix
            np.array(list(map(self.vocab, self.t_tokens)))[
                np.array(td) - 1
            ],  # when td is 0, it lies before our first breakpoint
        )
        new_times = np.insert(
            times, np.array(ix) + 1, np.array(times)[np.array(ix) + 1]
        ).astype(
            np.datetime64
        )  # spacing tokens assigned to the time at the end of the space
        return {"tokens": new_tokens, "times": new_times}

    def get_events_frame(self) -> Frame:
        """Combine all event tables into a single frame"""
        events = pl.concat(
            self.tbl[k].select("hadm_id", "event_time", "tokens", "times")
            for k in self.tbl.keys()
            if k not in ("patients", "admissions", "d_labitems", "d_items")
        )

        # Aggregate tokens and times by hadm_id
        tokens_agg = (
            events.lazy()
            .sort("event_time", pl.col("tokens").list.first())
            .group_by("hadm_id", maintain_order=True)
            .agg([pl.col("tokens").explode()])
        )

        times_agg = (
            events.lazy()
            .sort("event_time")
            .group_by("hadm_id", maintain_order=True)
            .agg([pl.col("times").explode()])
        )

        event_tokens = tokens_agg.join(
            times_agg, on="hadm_id", validate="1:1", maintain_order="left"
        )

        if self.include_time_spacing_tokens:
            event_tokens = event_tokens.with_columns(
                pl.struct(["tokens", "times"])
                .map_elements(
                    lambda x: self.time_spacing_inserter(x["tokens"], x["times"])[
                        "tokens"
                    ],
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("tokens"),
                pl.struct(["tokens", "times"])
                .map_elements(
                    lambda x: self.time_spacing_inserter(x["tokens"], x["times"])[
                        "times"
                    ],
                    return_dtype=pl.List(pl.Datetime(time_unit="ms")),
                )
                .alias("times"),
            )
        return event_tokens

    def cut_at_time(
        self, tokens_timelines: Frame, duration: pl.Duration = pl.duration(days=1)
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

    def get_tokens_timelines(self) -> Frame:
        """Main method to generate tokenized timelines using functional pipeline"""
        # Load tables
        self._load_tables()
        
        # Process tables functionally
        processed_tables = self._process_tables(self.tbl)
        self.tbl = processed_tables  # Update state
        
        # Apply quality control
        self.run_times_qc()
        
        # Apply quantization functionally
        quantized_tables = self._apply_quantization(self.tbl)
        self.tbl = quantized_tables  # Update state
        
        # Combine timelines functionally
        combined_timelines = self._combine_timelines(self.tbl)
        
        # Apply final filters
        return self._apply_filters(combined_timelines)

    def pad_and_truncate(self, tokens_timelines: Frame) -> Frame:
        """Pad and truncate token sequences to max_padded_length"""
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
        """Print auxiliary information from vocabulary"""
        self.vocab.print_aux()


def summarize(
    tokenizer: MimicTokenizer,
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


@functools.cache
def token_type(word: str) -> str:
    """Determine token type for a given word"""
    if word in MimicTokenizer().special:
        return "SPECIAL"
    elif re.fullmatch(r"Q\d", word) or re.fullmatch(r"Q[0-3][+-]", word):
        return "Q"
    else:
        return word.split("_")[0]


type_names = collections.OrderedDict(
    Q="Q",
    RACE="RACE",
    SEX="SEX",
    ADMN="ADMISSION",
    ADT="TRANSFER",
    LAB="LAB",
    VTL="VITALS",
    DSCG="DISCHARGE",
    SPECIAL="SPECIAL",
)
token_types = tuple(type_names.keys())


if __name__ == "__main__":
    # Example usage
    if os.uname().nodename.startswith("cri"):
        hm = pathlib.Path("/gpfs/data/bbj-lab/data/physionet.org/files/mimiciv_parquet")
    else:
        # change following line to develop locally
        hm = pathlib.Path("~/Documents/mimic-iv/3.1")

    out_dir = hm.parent.joinpath("mimic-tokenized").expanduser()
    out_dir.mkdir(exist_ok=True)

    tkzr = MimicTokenizer(
        data_dir=hm,
        max_padded_len=1024,
        day_stay_filter=True,
        valid_admission_window=("2110-01-01", "2111-12-31"),
        drop_nulls_nans=True,
        include_time_spacing_tokens=True,
    )
    tt = tokens_timelines = tkzr.get_tokens_timelines()

    tkzr.print_aux()
    summarize(tkzr, tokens_timelines)

    tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
    tokens_timelines.write_parquet(out_dir.joinpath("tokens_timelines.parquet"))
    tkzr.vocab.save(out_dir.joinpath("vocab.gzip"))

    # Test loading
    tkzr2 = MimicTokenizer(data_dir=hm, vocab_path=out_dir.joinpath("vocab.gzip"))
    tokens_timelines2 = tkzr2.get_tokens_timelines()
    assert len(tkzr.vocab) == len(tkzr2.vocab)
    assert tkzr.vocab.lookup == tkzr2.vocab.lookup
