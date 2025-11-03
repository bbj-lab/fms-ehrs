#!/usr/bin/env python3

"""
MIMIC-IV data processor that only handles data loading and basic formatting.
Returns raw data for the tokenizer to process.
"""

import os
import pathlib
import typing

import polars as pl

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class MIMICIVDataProcessor:
    """
    Data processor for MIMIC-IV dataset.
    Only handles data loading and basic formatting - no tokenization.
    """

    def __init__(self, data_dir: Pathlike, limit: int = None):
        """
        Initialize the MIMIC-IV data processor.

        Args:
            data_dir: Directory containing MIMIC-IV parquet files
            limit: Number of admission records to process. If None or -1, process all records.
        """
        self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
        self.hosp_dir = self.data_dir / "hosp"
        self.icu_dir = self.data_dir / "icu"
        self.limit = limit

    def get_reference_query(self) -> pl.LazyFrame:
        """Get the reference frame query with static patient data."""
        # Load main admissions table
        df = pl.scan_parquet(self.hosp_dir / "admissions.parquet")

        # Limit the number of admission records used if limit is specified and not -1
        if self.limit is not None and self.limit != -1:
            df = df.limit(self.limit)

        # Join patient demographics
        df = df.join(
            pl.scan_parquet(self.hosp_dir / "patients.parquet"),
            on="subject_id",
            validate="m:1",
            how="left",
        )

        return df

    def get_prefix_query(self) -> pl.LazyFrame:
        """
        Get raw prefix data query (demographics, admission info) without tokenization.

        Returns:
            LazyFrame with columns: subject_id, hadm_id, admission_time,
            race, sex, age_at_admission,
            admission_type
        """
        df = self.get_reference_query()

        prefix_query = df.select(
            pl.col("subject_id").cast(
                pl.Utf8
            ),  # This will use the left subject_id from admissions
            pl.col("hadm_id").cast(pl.Utf8),
            pl.col("admittime")
            .alias("admission_time")
            .cast(pl.Datetime(time_unit="ms")),
            pl.col("race"),
            pl.col("gender").alias("sex"),
            pl.col("anchor_age").alias("age_at_admission"),
            pl.col("admission_type").alias("admission_type"),
        )

        # print('Debug prefix query')
        # print(prefix_query.head(10).collect())

        return prefix_query

    def get_event_query(self, event_config: dict) -> pl.LazyFrame:
        """
        Get raw event data query for a specific event type without tokenization.

        Args:
            event_config: Event configuration from config file

        Returns:
            LazyFrame with columns: subject_id, event_time, category, value
        """
        table = event_config["table"]
        time_col = event_config["time"]
        code_col = event_config["code"]
        numeric_value_col = event_config.get("numeric_value")
        text_value_col = event_config.get("text_value")
        # filter_expr = event_config.get("filter")

        # Determine if it's a hospital or ICU table
        if table.startswith("icu/"):
            table_path = self.icu_dir / f"{table.split('/')[1]}.parquet"
        else:
            table_path = self.hosp_dir / f"{table}.parquet"

        # print(f"   Table path: {table_path}")
        # print(f"   Table exists: {table_path.exists()}")

        # Load the event table
        df = pl.scan_parquet(table_path)

        # Apply hospitalization limit if specified and not -1
        if self.limit is not None and self.limit != -1:
            # Get the same limited set of hospitalizations from admissions
            limited_hospitalizations = (
                pl.scan_parquet(self.hosp_dir / "admissions.parquet")
                .select("subject_id", "hadm_id")  # Include hadm_id to be more explicit
                .limit(self.limit)
            )
            # Filter events to only include the limited hospitalizations
            # Join on BOTH subject_id AND hadm_id to avoid Cartesian product
            df = df.join(
                limited_hospitalizations,
                on=[
                    "subject_id",
                    "hadm_id",
                ],  # Join on both columns to avoid duplicates
                how="inner",
            )

        # Apply filter if specified
        # if filter_expr:
        #     df = df.filter(eval(filter_expr))

        # Select relevant columns
        select_cols = [
            pl.col("subject_id").cast(
                pl.Utf8
            ),  # This will use the left subject_id from the event table
            pl.col("hadm_id").cast(
                pl.Utf8
            ),  # Use hadm_id from the event table (most MIMIC-IV tables have this)
        ]

        # Add time column - always required for timeline construction
        if time_col is None:
            raise ValueError(f"Time column is required for event table {table}")

        select_cols.append(
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias("time")
        )

        # DEBUG: Check datetime precision
        # print(f"DEBUG DATETIME: Table {table}, time_col: {time_col}")
        # print(f"DEBUG DATETIME: Original column dtype: {df.select(pl.col(time_col)).dtypes}")
        # print(f"DEBUG DATETIME: After cast to ms: {df.select(pl.col(time_col).cast(pl.Datetime(time_unit="ms"))).dtypes}")

        # CLIF codes are all string categories such as albumin, hemoglobin, ketamine etc.,
        # but raw mimiciv codes may just be integer itemids and our tokenizer expects
        # all codes to be strings so we need to cast itemids to strings.
        select_cols.append(
            pl.col(code_col)
            .cast(pl.Utf8)  # Cast to string first
            .str.to_lowercase()
            .str.replace_all(" ", "_")
            .str.strip_chars(".")
            .alias("code")
        )

        # Filter out events without hadm_id for tables that require it
        if table in ["labevents", "microbiologyevents"]:
            df = df.filter(pl.col("hadm_id").is_not_null())

        # Handle numeric and text values intelligently
        if table == "omr":
            # For OMR table, treat all values as text (observational data)
            select_cols.extend(
                [
                    pl.lit(None).alias("numeric_value"),  # No numeric values for OMR
                    pl.col("result_value").alias("text_value"),  # All values as text
                ]
            )
        elif table == "labevents":
            # For labevents, use valuenum for numeric, value for text (but only if value is not numeric)
            select_cols.extend(
                [
                    pl.col("valuenum").alias("numeric_value"),
                    pl.when(pl.col("value").str.contains(r"^\d+\.?\d*$"))
                    .then(None)
                    .otherwise(pl.col("value"))
                    .alias("text_value"),
                ]
            )
        else:
            # For other tables, use the original logic
            if numeric_value_col:
                select_cols.append(pl.col(numeric_value_col).alias("numeric_value"))

            if text_value_col:
                select_cols.append(pl.col(text_value_col).alias("text_value"))

        event_query = df.select(select_cols)

        # DEBUG: Check final datetime precision
        # print(f"DEBUG DATETIME: Final event_query time column dtype: {event_query.select(pl.col('time')).dtypes}")
        # print(f"DEBUG DATETIME: Sample time values: {event_query.select(pl.col('time')).head(3).collect()}")

        # print(f'Debug event query, table {table}')
        # print(event_query.head(10).collect())

        return event_query

    def get_suffix_query(self) -> pl.LazyFrame:
        """
        Get raw suffix data query (discharge info) without tokenization.

        Returns:
            LazyFrame with columns: subject_id, hadm_id, discharge_time,
            discharge_category
        """
        df = self.get_reference_query()

        suffix_query = df.select(
            pl.col("subject_id").cast(
                pl.Utf8
            ),  # This will use the left subject_id from admissions
            pl.col("hadm_id").cast(pl.Utf8),
            pl.col("dischtime")
            .alias("discharge_time")
            .cast(pl.Datetime(time_unit="ms")),
            pl.col("discharge_location").alias("discharge_category"),
        )

        # print('Debug suffix query')
        # print(suffix_query.head(10).collect())

        return suffix_query
