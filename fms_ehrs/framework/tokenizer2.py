#!/usr/bin/env python3

"""
A completely independent generic tokenizer that can tokenize any data as long as it's provided in a specific format.
This separates the tokenization process from data processing and doesn't rely on BaseTokenizer.
"""

import logging
import os
import pathlib
import re
import typing

import numpy as np
import pandas as pd
import polars as pl

import ruamel.yaml as yaml

from fms_ehrs.framework.vocabulary import Vocabulary

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class Tokenizer:
    """
    A completely independent tokenizer that implements all necessary functionality
    without relying on BaseTokenizer.
    """
    
    def __init__(
        self,
        config_file: Pathlike,
        *,
        vocab_path: Pathlike = None,
        max_padded_len: int = None,
        quantizer: typing.Literal["deciles", "sigmas"] = None,
        include_time_spacing_tokens: bool = None,
        cut_at_24h: bool = None,
        day_stay_filter: bool = None,
    ):
        """
        Initialize tokenizer.
        
        Args:
            config_file: Path to tokenizer configuration file
            vocab_path: Path to existing vocabulary file (for inference mode)
            max_padded_len: Maximum length for padding/truncation (overrides config)
            quantizer: Quantization method for numerical values (overrides config)
            include_time_spacing_tokens: Whether to include time spacing tokens (overrides config)
            cut_at_24h: Whether to cut timelines at 24 hours (overrides config)
            day_stay_filter: Whether to filter for day stays only (overrides config)
        """
        # Load configuration
        self.config = yaml.YAML(typ="safe").load(
            pathlib.Path(config_file).expanduser().resolve()
        )
        
        # Use config values with parameter overrides
        self.quantizer = quantizer if quantizer is not None else self.config["tokenizer"]["quantizer"]
        self.q_tokens = (
            tuple(map(lambda i: f"Q{i}", range(10)))
            if self.quantizer == "deciles"
            else ("Q3-", "Q2-", "Q1-", "Q0-", "Q0+", "Q1+", "Q2+", "Q3+")
        )
        self.special: tuple = ("TL_START", "TL_END", "PAD", "TRUNC", None, "nan")
        self.max_padded_length = max_padded_len if max_padded_len is not None else self.config["tokenizer"]["max_padded_len"]
        self.include_time_spacing_tokens = include_time_spacing_tokens if include_time_spacing_tokens is not None else self.config["tokenizer"]["include_time_spacing_tokens"]
        
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
        
        # Initialize vocabulary
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
        
        self.cut_at_24h = cut_at_24h if cut_at_24h is not None else self.config["tokenizer"]["cut_at_24h"]
        self.day_stay_filter = day_stay_filter if day_stay_filter is not None else self.config["tokenizer"]["day_stay_filter"]
    
    def set_quants(self, v: np.array, c: str, label: str = None) -> None:
        """Store training quantile information in the self.vocab object"""
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
        """Obtain corresponding quantiles using self.vocab object"""
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
    
    # def process_cat_val_frame(self, df: Frame, label: str) -> pl.LazyFrame:
    #     """Process event data by grouping by code (item ID) and generating [prefix_code, quantile] tokens."""
    #     # Use group_by with map_groups to process each code group lazily by only collecting the data
    #     # for each group as needed.
    #     # This avoids collecting the entire dataset while still processing groups individually
    #     return (
    #         df.group_by("code", maintain_order=True)
    #         .map_groups(
    #             lambda group: self._process_single_category(group, label, "time"),
    #             schema={
    #                 "subject_id": pl.Utf8,
    #                 "time": pl.Datetime(time_unit="ms"),
    #                 "code": pl.Utf8,
    #                 "numeric_value": pl.Float64,
    #                 "text_value": pl.Utf8,
    #                 "tokens": pl.List(pl.Int64),
    #                 "times": pl.List(pl.Datetime(time_unit="ms"))
    #             }
    #         )
    #     )      
    
    def process_cat_val_frame_with_text(self, df: Frame, label: str) -> pl.LazyFrame:
        """Process event data by grouping by code (item ID) and generating [prefix_code, quantile, text] tokens."""
        # Group by code to compute quantiles
        return (
            df.group_by("code", maintain_order=True)
            .map_groups(
                lambda group: self._process_single_category_with_text(group, label, "time"),
                schema={
                    "subject_id": pl.Utf8,
                    "hadm_id": pl.Utf8,
                    "time": pl.Datetime(time_unit="ms"),
                    "code": pl.Utf8,
                    "tokens": pl.List(pl.Int64),
                    "times": pl.List(pl.Datetime(time_unit="ms"))
                }
            )
        )
    
    def process_categorical_value(self, df: Frame, category_col: str, value_col: str, 
                                label: str, time_col: str = "event_time") -> Frame:
        """Process a dataframe with categorical values and numerical values."""
        return pl.concat(
            self._process_single_category(x, label, time_col) 
            for x in df.partition_by(category_col)
        )
    
    def _process_single_category(self, x: Frame, label: str, time_col: str) -> Frame:
        """Process a single category group."""
        v = x.select("numeric_value").to_numpy().ravel()
        c = x.select("code").row(0)[0]
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
                times=pl.concat_list(time_col, time_col).cast(
                    pl.List(pl.Datetime(time_unit="ms"))
                ),
            )
        )
    
    def _process_single_category_with_text(self, x: Frame, label: str, time_col: str) -> Frame:
        """Process a single category group with both numeric and text values."""
        
        # Debug: Check what type of object we're getting

        # Ensure that ms precision is used for time.
        # This is necessary even though the data processor already casts all Datetimes to 'ms' precision
        # because when polars converts a python list to a polars list column, it infers the precision from the individual datetime values 
        # rather than preserving the original precision.
        x = x.with_columns(
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias(time_col)
        )
        
        # Get numeric values for quantile computation
        v = x["numeric_value"].to_numpy().ravel()
        c = x["code"][0]
        
        # Get max text length from config
        max_text_length = self.config.get("max_text_value_length", 10)
        
        # Set quantiles for numeric values
        self.set_quants(v=v, c=c, label=label)
        
        # Get quantiles for the entire array once (outside the loop)
        # Use the same logic as get_quants but return numpy array directly
        designator = f"{label}_{c}" if label is not None else c
        if self.vocab.has_aux(designator):
            quantile_values = np.where(
                np.isfinite(v),
                np.digitize(v, bins=self.vocab.get_aux(designator)),
                self.vocab("nan"),
            )
        else:
            quantile_values = np.full(len(v), self.vocab(None))
        
        # Process each row
        tokens_list = []
        times_list = []
        
        for i in range(len(x)):
            row_tokens = []
            row_times = []
            
            
            # Always add code token
            code_token = self.vocab(f"{label}_{c}")
            if code_token not in [self.vocab(None), self.vocab("nan")]:
                row_tokens.append(code_token)
                row_times.append(x[time_col][i])
            
            # Add quantile token if numeric value exists
            try:
                numeric_val = x["numeric_value"][i]
            except Exception as e:
                print(f"DEBUG INDEXING ERROR: i={i}, x type={type(x)}, x['numeric_value'] type={type(x['numeric_value'])}")
                print(f"DEBUG INDEXING ERROR: x shape={x.shape}, x columns={x.columns}")
                print(f"DEBUG INDEXING ERROR: Exception={e}")
                raise e
            if numeric_val is not None and not np.isnan(numeric_val):
                quantile_token = quantile_values[i]
                if quantile_token not in [self.vocab(None), self.vocab("nan")]:
                    row_tokens.append(quantile_token)
                    row_times.append(x[time_col][i])
            
            # Add text value if it exists and passes filters
            text_val = x["text_value"][i]
            if text_val is not None and str(text_val).strip():
                text_str = str(text_val).strip()
                
                # First, try to cast to numbers - if it's a number, skip it
                try:
                    float(text_str)
                    text_str = None  # This is a number, skip it
                except ValueError:
                    # This is not a number, proceed with text processing
                    
                    # Clean text: remove spaces and keep only alphanumeric characters
                    text_str = re.sub(r'[^a-zA-Z0-9]', '', text_str)
                    
                    # Filter out text values that are too long
                    if len(text_str) > max_text_length:
                        text_str = None
                    
                    # Filter out empty strings after cleaning
                    if text_str is not None and len(text_str) == 0:
                        text_str = None
                
                # Add text value as a token if it passed all filters
                if text_str is not None:
                    text_token = self.vocab(f"TEXT_{text_str}")
                    if text_token not in [self.vocab(None), self.vocab("nan")]:
                        row_tokens.append(text_token)
                        row_times.append(x[time_col][i])
            
            # Only add if we have at least one token (code token should always be there)
            if row_tokens:
                tokens_list.append(row_tokens)
                times_list.append(row_times)
        
        # Convert to DataFrame
        if tokens_list:
            result = pl.DataFrame({
                "subject_id": x["subject_id"].to_list(),
                "hadm_id": x["hadm_id"].to_list(),
                "time": x[time_col].to_list(),
                "code": [c] * len(tokens_list),
                "tokens": tokens_list,
                "times": times_list
            }).with_columns(
                pl.col("times").cast(pl.List(pl.Datetime(time_unit="ms")))
            )
            return result
        else:
            # Return empty DataFrame with correct schema
            print("DEBUG: No tokens generated, returning empty DataFrame")
            return pl.DataFrame({
                "subject_id": [],
                "hadm_id": [],
                "time": [],
                "code": [],
                "tokens": [],
                "times": []
            }, schema={
                "subject_id": pl.Utf8,
                "hadm_id": pl.Utf8,
                "time": pl.Datetime(time_unit="ms"),
                "code": pl.Utf8,
                "tokens": pl.List(pl.Int64),
                "times": pl.List(pl.Datetime(time_unit="ms"))
            })
    
    def time_spacing_inserter(self, tokens, times):
        """Insert time spacing tokens between events based on time differences."""
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
        """Allows us to select the first 24h of someone's timeline for predictive purposes"""
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
        """Pad short sequences and truncate long sequences to max_padded_length"""
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
    
    def process_prefix_data(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Process prefix data into tokenized format using config.
        
        Args:
            data: LazyFrame with columns: subject_id, hadm_id, admission_time, 
                     race, ethnicity, sex, age_at_admission, 
                     admission_type
                     
        Returns:
            LazyFrame with columns: subject_id, hadm_id, prefix_tokens, prefix_times
        """
        # Process each prefix column according to config
        all_tokens = []
        all_times = []
        
        # Add TL_START token
        all_tokens.append(pl.lit(self.vocab("TL_START")))
        all_times.append(pl.col("admission_time"))
        
        for col_config in self.config["prefix"]:
            column = col_config["column"]
            
            if col_config.get("quantize", False):
                # Handle quantization for numerical columns (like age_at_admission)
                # For lazy evaluation, we need to collect just this column for quantization
                # This is a trade-off - we collect only what we need for quantile computation
                values = data.select(column).collect().to_numpy().ravel()
                self.set_quants(v=values, c=column.upper())
                quantized = self.get_quants(v=values, c=column.upper())
                all_tokens.append(quantized)
            else:
                # Handle string columns with prefixes
                prefix = col_config["prefix"]
                formatted_values = (
                    pl.col(column)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_")
                    .map_elements(
                        lambda x, p=prefix: self.vocab(f"{p}_{x}"),
                        return_dtype=pl.Int64,
                        skip_nulls=True,
                    )
                )
                all_tokens.append(formatted_values)
            
            # Add corresponding time (admission time for all prefix tokens)
            all_times.append(pl.col("admission_time"))
        
        # Combine all tokens and times
        # We need to include all columns that are referenced in the expressions
        return data.with_columns(
            prefix_tokens=pl.concat_list(all_tokens),
            prefix_times=pl.concat_list(all_times)
        ).select("subject_id", "hadm_id", "admission_time", "prefix_tokens", "prefix_times")
    
    def get_event(self, data_processor, event_config: dict) -> pl.LazyFrame:
        """
        Process a single event table following the tokenizer21.py pattern.
        
        Args:
            data_processor: Data processor with get_event_query method
            event_config: Event configuration from config file
            
        Returns:
            DataFrame with columns: hospitalization_id, time, tokens, times
        """
        print(f"\n🔍 DEBUG get_event - Processing: {event_config.get('table', 'UNKNOWN')}")
        # Get the lazy query for this event
        df = data_processor.get_event_query(event_config)

        # Debug: Show what the raw data looks like
        print("Raw event data (first 5 rows):")
        print(df.head(5).collect())
        
        prefix = event_config["prefix"]
        
        # Check if this event has numeric_value or text_value columns (categorical-value pairs)
        if "numeric_value" in df.collect_schema().names() or "text_value" in df.collect_schema().names():
            # Process as categorical-value pairs with numeric and/or text values
            print("Processing as categorical-value pairs with numeric and/or text values")
            
            # Special debugging for lab events
            if event_config.get("table") == "labevents":
                print("DEBUG LABEVENTS: About to process lab events data")
                print(f"DEBUG LABEVENTS: df type: {type(df)}")
                print(f"DEBUG LABEVENTS: df columns: {df.collect_schema().names()}")
            
            result = self.process_cat_val_frame_with_text(
                df,
                label=prefix,
            ).select("subject_id", "hadm_id", "tokens", "times")

            # Debug: Show what the processed result looks like
            print("Processed result (first 5 rows):")
            print(result.head(5).collect())

            return result
        else:
            # Process as simple categorical events (only code token)
            print("Processing as simple categorical events")
            
            result = df.select(
                pl.col("subject_id"),
                pl.col("hadm_id"),
                pl.concat_list([
                    pl.col("code").map_elements(
                        lambda x, prefix=prefix: self.vocab(f"{prefix}_{x}"),
                        return_dtype=pl.Int64,
                        skip_nulls=True,
                    )
                ]).alias("tokens"),
                pl.concat_list([
                    pl.col("time")
                ]).alias("times"),
            )
            
            print(f"DEBUG: Simple categorical result schema: {result.collect_schema()}")
            print(f"DEBUG: Simple categorical result columns: {result.collect_schema().names()}")
            
            print("Processed result (first 5 rows):")
            print(result.head(5).collect())
            return result


    
    # def get_events(self, data_processor, event_config: dict) -> pl.LazyFrame:



    def process_suffix_data(self, raw_data: Frame) -> Frame:
        """
        Process raw suffix data into tokenized format using config.
        
        Args:
            raw_data: LazyFrame with columns: subject_id, hadm_id, discharge_time, 
                     discharge_category
                     
        Returns:
            LazyFrame with columns: subject_id, hadm_id, suffix_tokens, suffix_times
        """
        # Start with the base data - select all columns we need
        result = raw_data
        
        # Process each suffix column according to config
        all_tokens = []
        all_times = []
        
        for col_config in self.config["suffix"]:
            column = col_config["column"]
            prefix = col_config["prefix"]
            
            # Process string columns with prefixes
            formatted_values = (
                pl.col(column)
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .map_elements(
                    lambda x, p=prefix: self.vocab(f"{p}_{x}") if x and x.strip() else self.vocab("DSCG_UNKNOWN"),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                )
            )
            all_tokens.append(formatted_values)
            
            # Add corresponding time (discharge time for all suffix tokens)
            all_times.append(pl.col("discharge_time"))
        
        # Add TL_END token (use lazy expression instead of len)
        tl_end_tokens = pl.lit(self.vocab("TL_END"))
        tl_end_times = pl.col("discharge_time")
        
        all_tokens.append(tl_end_tokens)
        all_times.append(tl_end_times)
        
        # Combine all tokens and times
        return result.with_columns(
            suffix_tokens=pl.concat_list(all_tokens),
            suffix_times=pl.concat_list(all_times)
        ).select("subject_id", "hadm_id", "suffix_tokens", "suffix_times")
    
    def get_tokens_timelines(self, data_processor) -> Frame:
        """
        Generate tokenized timelines by combining prefix, events, and suffix tokens.
        
        This method demonstrates the complete refactored architecture:
        - Data processor loads raw data queries (get_prefix_query, get_event_query, get_suffix_query)
        - Tokenizer processes data using config-driven rules (process_prefix_data, process_event_data, process_suffix_data)
        - Results are combined into complete tokenized timelines
        
        Args:
            data_processor: An object with methods:
                - get_prefix_query() -> raw prefix data query
                - get_event_query(event_config) -> raw event data query for specific event type
                - get_suffix_query() -> raw suffix data query
                
        Returns:
            DataFrame with columns: hospitalization_id, tokens, times
        """
        # Process prefix tables
        prefix_query = data_processor.get_prefix_query()
        prefix_tokens = self.process_prefix_data(prefix_query)

        print('debug prefix')
        print(prefix_query.head(10).collect())

        # Process event tables
        all_event_tokens = []
        # for event_config in self.config["events"]:
        #     event_tokens = self.get_event(data_processor, event_config)
        #     all_event_tokens.append(event_tokens)
        for event_config in self.config["events"]:
            try:
                event_tokens = self.get_event(data_processor, event_config)
                if event_tokens is not None:
                    all_event_tokens.append(event_tokens)
                else:
                    print(f"ERROR: get_event returned None for {event_config.get('table', 'UNKNOWN')}")
            except Exception as e:
                print(f"ERROR: get_event failed for {event_config.get('table', 'UNKNOWN')}: {e}")
                # Skip this event entirely - don't append None
       
        # Combine all events 
        if all_event_tokens:
            # Concatenate all event tokens and times for each subject
            events = (
                pl.concat(all_event_tokens)
                .sort(pl.col("times").list.first())
                .group_by("hadm_id", maintain_order=True)
                .agg(
                    tokens=pl.col("tokens").explode(),
                    times=pl.col("times").explode()
                )
            )
            
            # Apply time spacing tokens if enabled
            if self.include_time_spacing_tokens:
                events = events.with_columns(
                    pl.struct(["tokens", "times"])
                    .map_elements(
                        lambda x: self.time_spacing_inserter(x["tokens"], x["times"])["tokens"],
                        return_dtype=pl.List(pl.Int64),
                    ).alias("tokens"),
                    pl.struct(["tokens", "times"])
                    .map_elements(
                        lambda x: self.time_spacing_inserter(x["tokens"], x["times"])["times"],
                        return_dtype=pl.List(pl.Datetime(time_unit="ms")),
                    ).alias("times"),
                )
        else:
            # No events - create empty events frame
            events = prefix_tokens.select("subject_id").with_columns(
                tokens=pl.lit([]).cast(pl.List(pl.Int64)),
                times=pl.lit([]).cast(pl.List(pl.Datetime(time_unit="ms")))
            )

        # print('debug events')
        # print(events.limit(10).collect())

        # Process suffix
        suffix_query = data_processor.get_suffix_query()  
        suffix_tokens = self.process_suffix_data(suffix_query)
        print('debug suffix tokens')
        print(suffix_tokens.limit(10).collect())
        
        # Debug: Check for duplicate hadm_ids
        # print("DEBUG: Checking for duplicate hadm_ids...")
        # prefix_duplicates = prefix_tokens.group_by("hadm_id").len().filter(pl.col("len") > 1).collect()
        # if prefix_duplicates.height > 0:
        #     print(f"DEBUG: Found {prefix_duplicates.height} duplicate hadm_ids in prefix_tokens")
        #     print(prefix_duplicates.head())
        
        # events_duplicates = events.group_by("hadm_id").len().filter(pl.col("len") > 1).collect()
        # if events_duplicates.height > 0:
        #     print(f"DEBUG: Found {events_duplicates.height} duplicate hadm_ids in events")
        #     print(events_duplicates.head())
        
        # suffix_duplicates = suffix_tokens.group_by("hadm_id").len().filter(pl.col("len") > 1).collect()
        # if suffix_duplicates.height > 0:
        #     print(f"DEBUG: Found {suffix_duplicates.height} duplicate hadm_ids in suffix_tokens")
        #     print(suffix_duplicates.head())

        # Combine all components
        tt = (
            prefix_tokens
            .join(
                events,
                on="hadm_id",
                how="left",
                validate="1:1",
            )
            .join(
                suffix_tokens,
                on="hadm_id",
                how="left",
                validate="1:1",
            )
            .with_columns(
                tokens=pl.concat_list("prefix_tokens", "tokens", "suffix_tokens"),
                times=pl.concat_list("prefix_times", "times", "suffix_times"),
                # tokens=pl.concat_list("prefix_tokens", "tokens"),
                # times=pl.concat_list("prefix_times", "times"),
            )
            .select("subject_id", "hadm_id", "tokens", "times")
            .sort(by="hadm_id")
        )
        
        # Apply filters
        # if self.day_stay_filter:
        #     tt = tt.filter(
        #         (pl.col("times").list.get(-1) - pl.col("times").list.get(0))
        #         >= pl.duration(days=1)
        #     )
        
        # if self.cut_at_24h:
        #     tt = self.cut_at_time(tt)
        
        # print('Collecting final results')
        # return tt.collect()

        print('Writing final results to disk (streaming)')
        # Write results to disk using streaming approach
        output_path = "mimiciv_timelines.parquet"
        
        # Use sink_parquet for streaming write without collecting in memory
        tt.sink_parquet(output_path)
        print(f"Results written to {output_path}")
        
        # Return a lazy reference to the parquet file for streaming operations
        return pl.scan_parquet(output_path)
    
    def print_aux(self) -> None:
        """Print auxiliary data (quantile information)"""
        self.vocab.print_aux()


def summarize(
    tokenizer: Tokenizer,
    tokens_timelines: pl.LazyFrame,
    k: int = 20,
    logger: logging.Logger = None,
) -> None:
    """Provide posthoc summary statistics"""
    
    post = logger.info if logger is not None else print

    post("Timelines generated: {}".format(tokens_timelines.select(pl.len()).collect().item()))
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
                    for t in tokens_timelines.sample(1, seed=s).select("tokens").collect().item()
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


if __name__ == "__main__":
    """
    Example usage of the tokenizer with CLIF data processor.
    This demonstrates how to use the new architecture.
    """
    import sys
    sys.path.append('..')
    from preprocessing.clif_data_processor import CLIFDataProcessor
    
    
    # Configuration
    data_dir = "/gpfs/data/bbj-lab/users/burkh4rt/development-sample/raw"
    config_file = "../config/config-tokenizer.yaml"
    
    # # Create data processor
    # clif_processor = CLIFDataProcessor(data_dir=data_dir)
    
    # # Create tokenizer
    # tokenizer = Tokenizer(config_file=config_file)
 
    # # Generate complete timelines
    # timelines = tokenizer.get_tokens_timelines(clif_processor)

    # # Generate summary statistics
    # summarize(tokenizer, timelines)

    # Example usage with MIMIC-IV data processor
    print("\n" + "="*50)
    print("MIMIC-IV Data Processor Example")
    print("="*50)
    
    # Import MIMIC-IV data processor
    from preprocessing.mimiciv_data_processor import MIMICIVDataProcessor
    
    # Configuration for MIMIC-IV
    mimiciv_data_dir = "/gpfs/data/bbj-lab/data/physionet.org/files/mimiciv_parquet"
    
    # Create MIMIC-IV data processor
    mimiciv_processor = MIMICIVDataProcessor(data_dir=mimiciv_data_dir, limit=1000)
    
    mimic_cfg = "../config/config-tokenizer-mimiciv.yaml"
    # Create tokenizer (using same config as CLIF for now)
    tokenizer_mimiciv = Tokenizer(config_file=mimic_cfg)
    
    # Generate complete timelines for MIMIC-IV data
    timelines_mimiciv = tokenizer_mimiciv.get_tokens_timelines(mimiciv_processor)

    # Generate summary statistics for MIMIC-IV
    summarize(tokenizer_mimiciv, timelines_mimiciv)
    
    # # Generate summary statistics for MIMIC-IV
    # print(f"\nMIMIC-IV Timeline Summary:")
    # print(f"Number of patients: {timelines_mimiciv.height}")
    # print(f"Average tokens per patient: {timelines_mimiciv.select(pl.col('tokens').list.len().mean()).item():.1f}")
    # print(f"Average timeline length (hours): {timelines_mimiciv.select(pl.col('times').list.len().mean()).item():.1f}")
    
    # # Show sample of MIMIC-IV data
    # print(f"\nSample MIMIC-IV timeline:")
    # sample = timelines_mimiciv.limit(1).collect()
    # if sample.height > 0:
    #     print(f"Patient ID: {sample['subject_id'][0]}")
    #     print(f"Number of tokens: {len(sample['tokens'][0])}")
    #     print(f"Timeline span: {sample['times'][0][0]} to {sample['times'][0][-1]}")
  
