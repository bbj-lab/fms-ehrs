#!/usr/bin/env python3

"""
A completely independent generic tokenizer that can tokenize any data as long as it's provided in a specific format.
This separates the tokenization process from data processing and doesn't rely on BaseTokenizer.
"""

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
        import ruamel.yaml as yaml
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
    
    def process_cat_val_frame(self, df: Frame, label: str) -> Frame:
        """Process a dataframe with categorical values and numerical values (from BaseTokenizer)."""
        return pl.concat(
            self._process_single_category(x, label, "event_time") 
            for x in df.partition_by("category")
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
                times=pl.concat_list(time_col, time_col).cast(
                    pl.List(pl.Datetime(time_unit="ms"))
                ),
            )
        )
    
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
    
    def process_prefix_data(self, raw_data: Frame) -> Frame:
        """
        Process raw prefix data into tokenized format using config.
        
        Args:
            raw_data: LazyFrame with columns: hospitalization_id, admission_dttm, 
                     race_category, ethnicity_category, sex_category, age_at_admission, 
                     admission_type_name
                     
        Returns:
            LazyFrame with columns: hospitalization_id, prefix_tokens, prefix_times
        """
        # Process each prefix column according to config
        all_tokens = []
        all_times = []
        
        # Add TL_START token
        all_tokens.append(pl.lit(self.vocab("TL_START")))
        all_times.append(pl.col("admission_dttm"))
        
        for col_config in self.config["prefix"]:
            column = col_config["column"]
            
            if col_config.get("quantize", False):
                # Handle quantization for numerical columns (like age_at_admission)
                # For lazy evaluation, we need to collect just this column for quantization
                # This is a trade-off - we collect only what we need for quantile computation
                values = raw_data.select(column).collect().to_numpy().ravel()
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
                        lambda x: self.vocab(f"{prefix}_{x}"),
                        return_dtype=pl.Int64,
                        skip_nulls=False,
                    )
                )
                all_tokens.append(formatted_values)
            
            # Add corresponding time (admission time for all prefix tokens)
            all_times.append(pl.col("admission_dttm"))
        
        # Combine all tokens and times
        # We need to include all columns that are referenced in the expressions
        return raw_data.with_columns(
            prefix_tokens=pl.concat_list(all_tokens),
            prefix_times=pl.concat_list(all_times)
        ).select("subject_id", "admission_dttm", "prefix_tokens", "prefix_times")
    
    def get_event(self, data_processor, event_config: dict) -> Frame:
        """
        Process a single event table following the tokenizer21.py pattern.
        
        Args:
            data_processor: Data processor with get_event_query method
            event_config: Event configuration from config file
            
        Returns:
            DataFrame with columns: hospitalization_id, event_time, tokens, times
        """
        # Get the lazy query for this event
        df = data_processor.get_event_query(event_config)
        
        prefix = event_config["prefix"]
        numeric_value_col = event_config.get("numeric_value")
        
        if numeric_value_col is not None:
            # Process as categorical-value pairs (like labs, vitals)
            # Collect the data and pass to process_cat_val_frame
            collected_data = df.collect()
            return self.process_cat_val_frame(
                collected_data,
                label=prefix,
            ).select("subject_id", "event_time", "tokens", "times")
        else:
            # Process as simple categorical events (like transfers, positions)
            # The data processor has already processed the code column(s) into "category"
            # and text_value column to "text_value" if it exists
            text_value_col = event_config.get("text_value")
            code_col = event_config["code"]
            
            # Determine if category is a list (multiple codes) or string (single code)
            is_list_category = isinstance(code_col, list)
            
            if text_value_col is not None:
                # Both category and text_value exist
                if is_list_category:
                    # Category is a list, text_value is a string
                    return df.select(
                        pl.col("subject_id"),
                        pl.col("event_time"),
                        pl.concat_list([
                            pl.col("category").list.eval(
                                pl.element().map_elements(
                                    lambda x, prefix=prefix: self.vocab(f"{prefix}_{x}"),
                                    return_dtype=pl.Int64,
                                    skip_nulls=False,
                                )
                            ),
                            pl.col("text_value")
                            .str.to_lowercase()
                            .str.replace_all(" ", "_")
                            .str.strip_chars(".")
                            .map_elements(
                                lambda x, prefix=prefix: self.vocab(f"{prefix}_{x}"),
                                return_dtype=pl.Int64,
                                skip_nulls=False,
                            )
                        ]).alias("tokens"),
                        pl.concat_list([
                            pl.col("event_time").repeat_by(pl.col("category").list.len()),
                            pl.col("event_time")
                        ]).alias("times"),
                    ).collect()
                else:
                    # Both category and text_value are strings
                    return df.select(
                        pl.col("subject_id"),
                        pl.col("event_time"),
                        pl.concat_list([
                            pl.col("category").map_elements(
                                lambda x, prefix=prefix: self.vocab(f"{prefix}_{x}"),
                                return_dtype=pl.Int64,
                                skip_nulls=False,
                            ),
                            pl.col("text_value")
                            .str.to_lowercase()
                            .str.replace_all(" ", "_")
                            .str.strip_chars(".")
                            .map_elements(
                                lambda x, prefix=prefix: self.vocab(f"{prefix}_{x}"),
                                return_dtype=pl.Int64,
                                skip_nulls=False,
                            )
                        ]).alias("tokens"),
                        pl.concat_list([
                            pl.col("event_time"),
                            pl.col("event_time")
                        ]).alias("times"),
                    ).collect()
            else:
                # Only category exists
                if is_list_category:
                    # Category is a list
                    return df.select(
                        pl.col("subject_id"),
                        pl.col("event_time"),
                        pl.col("category").list.eval(
                            pl.element().map_elements(
                                lambda x, prefix=prefix: self.vocab(f"{prefix}_{x}"),
                                return_dtype=pl.Int64,
                                skip_nulls=False,
                            )
                        ).alias("tokens"),
                        pl.col("event_time").repeat_by(pl.col("category").list.len()).alias("times"),
                    ).collect()
                else:
                    # Category is a string
                    return df.select(
                        pl.col("subject_id"),
                        pl.col("event_time"),
                        pl.concat_list([
                            pl.col("category").map_elements(
                                lambda x, prefix=prefix: self.vocab(f"{prefix}_{x}"),
                                return_dtype=pl.Int64,
                                skip_nulls=False,
                            )
                        ]).alias("tokens"),
                        pl.concat_list([
                            pl.col("event_time")
                        ]).alias("times"),
                    ).collect()
    
    
    def process_suffix_data(self, raw_data: Frame) -> Frame:
        """
        Process raw suffix data into tokenized format using config.
        
        Args:
            raw_data: LazyFrame with columns: hospitalization_id, discharge_dttm, 
                     discharge_category
                     
        Returns:
            LazyFrame with columns: hospitalization_id, suffix_tokens, suffix_times
        """
        # Start with the base data
        result = raw_data.select("subject_id")
        
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
                    lambda x: self.vocab(f"{prefix}_{x}"),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                )
            )
            all_tokens.append(formatted_values)
            
            # Add corresponding time (discharge time for all suffix tokens)
            all_times.append(pl.col("discharge_dttm"))
        
        # Add TL_END token (use lazy expression instead of len)
        tl_end_tokens = pl.lit(self.vocab("TL_END"))
        tl_end_times = pl.col("discharge_dttm")
        
        all_tokens.append(tl_end_tokens)
        all_times.append(tl_end_times)
        
        # Combine all tokens and times
        return result.with_columns(
            suffix_tokens=pl.concat_list(all_tokens),
            suffix_times=pl.concat_list(all_times)
        )
    
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
        # Get raw prefix data and process it with our new method
        prefix_query = data_processor.get_prefix_query()
        prefix_tokens = self.process_prefix_data(prefix_query)
        
        # Process all events using the tokenizer21.py pattern
        all_event_tokens = []
        for event_config in self.config["events"]:
            event_tokens = self.get_event(data_processor, event_config)
            all_event_tokens.append(event_tokens)
        
        # Combine all events following tokenizer21.py pattern
        if all_event_tokens:
            events = (
                pl.concat(all_event_tokens)
                .lazy()
                .sort("event_time", pl.col("tokens").list.first())
                .group_by("subject_id", maintain_order=True)
                .agg(tokens=pl.col("tokens").explode(), times=pl.col("times").explode())
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
        
        # Process suffix data using the new method
        suffix_query = data_processor.get_suffix_query()
        suffix_tokens = self.process_suffix_data(suffix_query)
        
        # Combine all components
        tt = (
            prefix_tokens
            .join(
                events,
                on="subject_id",
                how="left",
                validate="1:1",
            )
            .join(
                suffix_tokens,
                on="subject_id",
                how="left",
                validate="1:1",
            )
            .with_columns(
                tokens=pl.concat_list("prefix_tokens", "tokens", "suffix_tokens"),
                times=pl.concat_list("prefix_times", "times", "suffix_times"),
            )
            .select("subject_id", "tokens", "times")
            .sort(by="subject_id")
        )
        
        # Apply filters
        if self.day_stay_filter:
            tt = tt.filter(
                (pl.col("times").list.get(-1) - pl.col("times").list.get(0))
                >= pl.duration(days=1)
            )
        
        if self.cut_at_24h:
            tt = self.cut_at_time(tt)
        
        return tt.collect()
    
    def print_aux(self) -> None:
        """Print auxiliary data (quantile information)"""
        self.vocab.print_aux()


def summarize(
    tokenizer: Tokenizer,
    tokens_timelines: Frame,
    k: int = 20,
    logger: logging.Logger = None,
) -> None:
    """Provide posthoc summary statistics"""
    
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
    
    # Create data processor
    clif_processor = CLIFDataProcessor(data_dir=data_dir)
    
    # Create tokenizer
    tokenizer = Tokenizer(config_file=config_file)
 
    # Generate complete timelines
    timelines = tokenizer.get_tokens_timelines(clif_processor)

    # Generate summary statistics
    summarize(tokenizer, timelines)
  
