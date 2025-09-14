#!/usr/bin/env python3

"""
MIMIC-IV patients table loader and preprocessor
"""

import pathlib
import polars as pl
from typing import Union

Pathlike = Union[pathlib.PurePath, str]


def load_patients(table_path: Pathlike) -> pl.LazyFrame:
    """
    Load the patients table from parquet file.
    
    Args:
        table_path: Path to the patients.parquet file
        
    Returns:
        LazyFrame containing the patients data
    """
    # TODO: Implement loading logic
    pass


def preprocess_patients(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply CLIF-compatible preprocessing to patients table.
    
    This function applies the same data cleaning and preprocessing steps used in the 
    CLIF-MIMIC repository but keeps the data in raw MIMIC-IV format.
    
    Args:
        df: Raw patients LazyFrame
        
    Returns:
        Preprocessed patients LazyFrame (still in MIMIC format)
        
    Preprocessing steps (from CLIF-MIMIC):
    - Standardize ID column types (subject_id)
    - Handle missing values in anchor_age
    - Convert gender to standardized format
    - Apply data quality improvements used in CLIF preprocessing
    """
    # TODO: Implement CLIF-compatible preprocessing
    # Reference: https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-MIMIC
    pass
