#!/usr/bin/env python3

"""
CLIF-specific data preprocessing for tokenization.
Handles CLIF-2.1 standard data format and prepares it for the pure tokenizer.
"""

import os
import pathlib
import typing

import polars as pl
import ruamel.yaml as yaml

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class CLIFDataProcessor:
    """
    Data processor for CLIF-2.1 standard data format.
    Handles all CLIF-specific data loading and preprocessing.
    """

    def __init__(
        self,
        data_dir: Pathlike,
        config_file: Pathlike,
        tokenizer=None,  # Will be set by the tokenizer
    ):
        """
        Initialize the CLIF data processor.

        Args:
            data_dir: Directory containing CLIF parquet files
            config_file: Path to CLIF configuration file
            tokenizer: Reference to the tokenizer (for vocabulary access)
        """
        self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
        self.config = yaml.YAML(typ="safe").load(
            pathlib.Path(config_file).expanduser().resolve()
        )
        self.tokenizer = tokenizer  # Will be set by the tokenizer

    def get_reference_frame(self) -> Frame:
        """Get the reference frame with static patient data."""
        df = pl.scan_parquet(
            self.data_dir.joinpath(f"{self.config['reference']['table']}.parquet")
        )

        # Join augmentation tables
        for tkv in self.config["augmentation_tables"]:
            df = df.join(
                pl.scan_parquet(self.data_dir.joinpath(f"{tkv['table']}.parquet")),
                on=tkv["key"],
                validate=tkv["validation"],
                how="left",
            )

        # Process age if present
        if "age" in self.config["reference"]:
            age = (
                df.select(self.config["reference"]["age"]).collect().to_numpy().ravel()
            )
            if self.tokenizer:
                self.tokenizer.set_quants(v=age, c="AGE")
                df = df.with_columns(
                    quantized_age=self.tokenizer.get_quants(v=age, c="AGE")
                )

        return self._run_time_qc(df)

    def _run_time_qc(self, reference_frame) -> Frame:
        """Run time quality control if configured."""
        if "times_qc" not in self.config:
            return reference_frame

        return (
            reference_frame.join(
                pl.scan_parquet(
                    self.data_dir.joinpath(
                        f"{self.config['times_qc']['table']}.parquet"
                    )
                )
                .group_by(self.config["subject_id"])
                .agg(
                    start_time_alt=pl.col(self.config["times_qc"]["time"]).min(),
                    end_time_alt=pl.col(self.config["times_qc"]["time"]).max(),
                ),
                how="left",
                on=self.config["subject_id"],
                validate="1:1",
            )
            .with_columns(
                pl.min_horizontal(
                    self.config["reference"]["start_time"], "start_time_alt"
                )
                .cast(pl.Datetime(time_unit="ms"))
                .alias(self.config["reference"]["start_time"]),
                pl.max_horizontal(self.config["reference"]["end_time"], "end_time_alt")
                .cast(pl.Datetime(time_unit="ms"))
                .alias(self.config["reference"]["end_time"]),
            )
            .drop("start_time_alt", "end_time_alt")
            .filter(
                pl.col(self.config["reference"]["start_time"])
                < pl.col(self.config["reference"]["end_time"])
            )
        )

    def get_prefix_tokens(self) -> Frame:
        """Get prefix tokens (demographics, admission info, etc.)."""
        time_col = self.config["reference"]["start_time"]
        df = self.get_reference_frame()

        return df.select(
            pl.col(self.config["subject_id"]).alias("hospitalization_id"),
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
            pl.concat_list(
                [self.tokenizer.vocab("TL_START")]
                + [
                    (
                        pl.col(col["column"])
                        .str.to_lowercase()
                        .str.replace_all(" ", "_")
                        .map_elements(
                            lambda x, prefix=col["prefix"]: self.tokenizer.vocab(
                                f"{prefix}_{x}"
                            ),
                            return_dtype=pl.Int64,
                            skip_nulls=False,
                        )
                        if not col["column"].startswith("quantized")
                        else pl.col(col["column"])
                    )
                    for col in self.config["prefix"]
                ]
            ).alias("prefix_tokens"),
            pl.concat_list(
                [pl.col(time_col).cast(pl.Datetime(time_unit="ms"))]
                * (len(self.config["prefix"]) + 1)
            ).alias("prefix_times"),
        )

    def get_suffix_tokens(self) -> Frame:
        """Get suffix tokens (discharge info, etc.)."""
        time_col = self.config["reference"]["end_time"]
        df = self.get_reference_frame()

        return df.select(
            pl.col(self.config["subject_id"]).alias("hospitalization_id"),
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
            pl.concat_list(
                [
                    (
                        pl.col(col["column"])
                        .str.to_lowercase()
                        .str.replace_all(" ", "_")
                        .map_elements(
                            lambda x, prefix=col["prefix"]: self.tokenizer.vocab(
                                f"{prefix}_{x}"
                            ),
                            return_dtype=pl.Int64,
                            skip_nulls=False,
                        )
                        if not col["column"].startswith("quantized")
                        else pl.col(col["column"])
                    )
                    for col in self.config["suffix"]
                ]
                + [self.tokenizer.vocab("TL_END")]
            ).alias("suffix_tokens"),
            pl.concat_list(
                [pl.col(time_col).cast(pl.Datetime(time_unit="ms"))]
                * (len(self.config["suffix"]) + 1)
            ).alias("suffix_times"),
        )

    def get_events(self) -> Frame:
        """Get all event data (labs, vitals, medications, etc.) as tokenized events."""
        event_tokens = (
            pl.concat([self._process_event(evt) for evt in self.config["events"]])
            .lazy()
            .sort("event_time", pl.col("tokens").list.first())
        )

        # Group by hospitalization_id and aggregate
        event_tokens = (
            event_tokens.group_by("hospitalization_id", maintain_order=True)
            .agg(tokens=pl.col("tokens").explode(), times=pl.col("times").explode())
            .with_columns(
                pl.col("tokens").alias("event_tokens"),
                pl.col("times").alias("event_times"),
            )
        )

        # Add time spacing tokens if enabled
        if self.tokenizer and self.tokenizer.include_time_spacing_tokens:
            event_tokens = event_tokens.with_columns(
                pl.struct(["event_tokens", "event_times"])
                .map_elements(
                    lambda x: self.tokenizer.time_spacing_inserter(
                        x["event_tokens"], x["event_times"]
                    )["tokens"],
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("event_tokens"),
                pl.struct(["event_tokens", "event_times"])
                .map_elements(
                    lambda x: self.tokenizer.time_spacing_inserter(
                        x["event_tokens"], x["event_times"]
                    )["times"],
                    return_dtype=pl.List(pl.Datetime(time_unit="ms")),
                )
                .alias("event_times"),
            )

        return event_tokens

    def _process_event(self, event_config: dict) -> Frame:
        """Process a single event type from the configuration."""
        df = pl.scan_parquet(self.data_dir.joinpath(f"{event_config['table']}.parquet"))

        # Apply filter if specified
        if "filter" in event_config and event_config["filter"]:
            df = df.filter(eval(event_config["filter"]))

        if event_config.get("numeric_value"):
            # Process category-value events (labs, vitals, etc.)
            return self._process_category_value_event(df, event_config)
        else:
            # Process categorical-only events
            return self._process_categorical_event(df, event_config)

    def _process_category_value_event(self, df: Frame, event_config: dict) -> Frame:
        """Process events with both category and numeric values."""
        processed_df = df.select(
            pl.col(self.config["subject_id"]).alias("hospitalization_id"),
            pl.col(event_config["time"])
            .cast(pl.Datetime(time_unit="ms"))
            .alias("event_time"),
            pl.col(event_config["code"])
            .str.to_lowercase()
            .str.replace_all(" ", "_")
            .str.strip_chars(".")
            .alias("category"),
            pl.col(event_config["numeric_value"]).alias("value"),
        ).collect()

        # Use the tokenizer's category-value processing
        result = self.tokenizer.process_categorical_value(
            processed_df,
            category_col="category",
            value_col="value",
            label=event_config["prefix"],
            time_col="event_time",
        )

        return result.select("hospitalization_id", "event_time", "tokens", "times")

    def _process_categorical_event(self, df: Frame, event_config: dict) -> Frame:
        """Process events with only categorical values."""
        category_list = (
            [event_config["code"]]
            if isinstance(event_config["code"], str)
            else event_config["code"]
        ) + ([event_config["text_value"]] if event_config.get("text_value") else [])

        return df.select(
            pl.col(self.config["subject_id"]).alias("hospitalization_id"),
            pl.col(event_config["time"])
            .cast(pl.Datetime(time_unit="ms"))
            .alias("event_time"),
            pl.concat_list(
                [
                    pl.col(cat)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_")
                    .str.strip_chars(".")
                    .map_elements(
                        lambda x, prefix=event_config["prefix"]: self.tokenizer.vocab(
                            f"{prefix}_{x}"
                        ),
                        return_dtype=pl.Int64,
                        skip_nulls=False,
                    )
                    for cat in category_list
                ]
            ).alias("tokens"),
            pl.concat_list(
                [pl.col(event_config["time"]).cast(pl.Datetime("ms"))]
                * len(category_list)
            ).alias("times"),
        ).collect()
