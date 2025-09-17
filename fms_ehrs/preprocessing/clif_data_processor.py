#!/usr/bin/env python3

"""
CLIF data processor that only handles data loading and basic formatting.
Returns raw data for the tokenizer to process.
"""

import os
import pathlib
import typing

import polars as pl

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class CLIFDataProcessor:
    """
    Data processor for CLIF-2.1 standard data format.
    Only handles data loading and basic formatting - no tokenization.
    """
    
    def __init__(self, data_dir: Pathlike):
        """
        Initialize the CLIF data processor.
        
        Args:
            data_dir: Directory containing CLIF parquet files
        """
        self.data_dir = pathlib.Path(data_dir).expanduser().resolve()
    
    def get_reference_query(self) -> pl.LazyFrame:
        """Get the reference frame query with static patient data."""
        # Load main hospitalization table
        df = pl.scan_parquet(
            self.data_dir.joinpath("clif_hospitalization.parquet")
        )
        
        # Join patient demographics
        df = df.join(
            pl.scan_parquet(self.data_dir.joinpath("clif_patient.parquet")),
            on="patient_id",
            validate="m:1",
            how="left",
        )
        
        return df
    
    def get_prefix_query(self) -> pl.LazyFrame:
        """
        Get raw prefix data query (demographics, admission info) without tokenization.
        
        Returns:
            LazyFrame with columns: subject_id, admission_time, 
            race, ethnicity, sex, age_at_admission, 
            admission_type
        """
        df = self.get_reference_query()
        
        return df.select(
            pl.col("hospitalization_id").alias("subject_id"),
            pl.col("admission_dttm").alias("admission_time").cast(pl.Datetime(time_unit="ms")),
            pl.col("race_category").alias("race"),           
            pl.col("ethnicity_category").alias("ethnicity"),      
            pl.col("sex_category").alias("sex"),            
            "age_at_admission",
            pl.col("admission_type_name").alias("admission_type") 
        )
    
    def get_event_query(self, event_config: dict) -> pl.LazyFrame:
        """
        Get raw event data query for a specific event type without tokenization.
        
        Args:
            event_config: Event configuration from config file
            
        Returns:
            LazyFrame with columns: hospitalization_id, event_time, category, value
        """
        table = event_config["table"]
        time_col = event_config["time"]
        code_col = event_config["code"]
        numeric_value_col = event_config.get("numeric_value")
        text_value_col = event_config.get("text_value")
        filter_expr = event_config.get("filter")
        
        # Load the event table
        df = pl.scan_parquet(self.data_dir.joinpath(f"{table}.parquet"))
        
        # Apply filter if specified
        if filter_expr:
            df = df.filter(eval(filter_expr))
        
        # Select relevant columns
        select_cols = [
            pl.col("hospitalization_id").alias("subject_id"),
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias("time"),
        ]
        
        # Handle code column (can be string or list)
        if isinstance(code_col, str):
            select_cols.append(
                pl.col(code_col)
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .str.strip_chars(".")
                .alias("code")
            )
        else:  # list of codes
            select_cols.append(
                pl.concat_list([
                    pl.col(cat)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_")
                    .str.strip_chars(".")
                    for cat in code_col
                ]).alias("code")
            )
        
        # Add value column if numeric_value is specified
        if numeric_value_col:
            select_cols.append(pl.col(numeric_value_col).alias("numeric_value"))
        
        # Add text_value if specified
        if text_value_col:
            select_cols.append(pl.col(text_value_col).alias("text_value"))
        
        return df.select(select_cols)
    
    def get_suffix_query(self) -> pl.LazyFrame:
        """
        Get raw suffix data query (discharge info) without tokenization.
        
        Returns:
            LazyFrame with columns: subject_id, discharge_time, 
            discharge_category
        """
        df = self.get_reference_query()
        
        return df.select(
            pl.col("hospitalization_id").alias("subject_id"),
            pl.col("discharge_dttm").alias("discharge_time").cast(pl.Datetime(time_unit="ms")),
            "discharge_category"
        )
