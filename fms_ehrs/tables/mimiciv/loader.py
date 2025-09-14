#!/usr/bin/env python3

"""
Data loader for MIMIC-IV data with CLIF-compatible preprocessing
"""

import os
import pathlib
import typing
from typing import Optional, Dict, List

import polars as pl

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike

from .admissions import load_admissions, preprocess_admissions
from .chartevents import load_chartevents, preprocess_chartevents
from .d_items import load_d_items, preprocess_d_items
from .d_labitems import load_d_labitems, preprocess_d_labitems
from .icustays import load_icustays, preprocess_icustays
from .labevents import load_labevents, preprocess_labevents
from .patients import load_patients, preprocess_patients

# Default data path
MIMIC_PARQUET_PATH = "/gpfs/data/bbj-lab/data/physionet.org/files/mimiciv_parquet"

# Required tables organized by category
REQUIRED_TABLES = {
    "hosp": [
        "patients",
        "admissions", 
        "labevents",
        "d_labitems"
    ],
    "icu": [
        "icustays",
        "chartevents",
        "d_items"
    ]
}


class MimicDataLoader:
    """
    Data loader for MIMIC-IV data with CLIF-compatible preprocessing.
    
    This loader applies the same data preprocessing steps used in the 
    CLIF-MIMIC repository (data cleaning, type standardization, missing value 
    handling) but keeps the data in raw MIMIC-IV format rather than converting 
    to CLIF format. This ensures consistency for comparison studies between 
    raw MIMIC tokens vs CLIF tokens.
    """
    
    def __init__(
        self, 
        data_dir: Pathlike = None,
    ):
        """
        Initialize the MIMIC data loader.
        
        Args:
            data_dir: Path to MIMIC-IV parquet data directory
        """
        self.data_dir = pathlib.Path(data_dir).expanduser() if data_dir else pathlib.Path(MIMIC_PARQUET_PATH)
        self.tables: Dict[str, pl.LazyFrame] = {}
        
        # Set up directory paths
        self.hosp_dir = self.data_dir / "hosp"
        self.icu_dir = self.data_dir / "icu"
    
    def load_tables(self) -> None:
        """Load all required MIMIC-IV tables with preprocessing."""
        if not self.validate_data_structure():
            raise ValueError(f"Invalid MIMIC-IV parquet data structure in {self.data_dir}")
        
        # Load hospital tables
        self.load_hospital_tables()
        
        # Load ICU tables
        self.load_icu_tables()
    
    def load_hospital_tables(self) -> None:
        """Load hospital tables with CLIF-compatible preprocessing."""
        for table_name in REQUIRED_TABLES["hosp"]:
            self._load_table(table_name, "hosp")
    
    def load_icu_tables(self) -> None:
        """Load ICU tables with CLIF-compatible preprocessing."""
        for table_name in REQUIRED_TABLES["icu"]:
            self._load_table(table_name, "icu")
    
    def _load_table(self, table_name: str, category: str) -> None:
        """Load a single table with appropriate preprocessing."""
        # Get the appropriate directory
        data_dir = self.hosp_dir if category == "hosp" else self.icu_dir
        table_path = data_dir / f"{table_name}.parquet"
        
        if not table_path.exists():
            raise FileNotFoundError(f"Table file not found: {table_path}")
        
        # Load the table using table-specific loader
        df = self._load_table_by_name(table_name, table_path)
        
        # Apply preprocessing 
        df = self._preprocess_table_by_name(table_name, df)
        
        self.tables[table_name] = df
        print(f"✓ Loaded table: {table_name}")
    
    def _load_table_by_name(self, table_name: str, table_path: pathlib.Path) -> pl.LazyFrame:
        """Load table using table-specific loader function."""
        loaders = {
            "patients": load_patients,
            "admissions": load_admissions,
            "labevents": load_labevents,
            "d_labitems": load_d_labitems,
            "icustays": load_icustays,
            "chartevents": load_chartevents,
            "d_items": load_d_items,
        }
        
        if table_name not in loaders:
            raise ValueError(f"Unknown table: {table_name}")
        
        return loaders[table_name](table_path)
    
    def _preprocess_table_by_name(self, table_name: str, df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply CLIF-compatible preprocessing using table-specific function."""
        preprocessors = {
            "patients": preprocess_patients,
            "admissions": preprocess_admissions,
            "labevents": preprocess_labevents,
            "d_labitems": preprocess_d_labitems,
            "icustays": preprocess_icustays,
            "chartevents": preprocess_chartevents,
            "d_items": preprocess_d_items,
        }
        
        if table_name not in preprocessors:
            raise ValueError(f"Unknown table: {table_name}")
        
        return preprocessors[table_name](df)
    
    def validate_data_structure(self) -> bool:
        """Validate MIMIC-IV parquet directory structure."""
        if not self.data_dir.exists():
            return False
        
        if not self.hosp_dir.exists() or not self.icu_dir.exists():
            return False
        
        # Check all required table files exist
        for category, table_names in REQUIRED_TABLES.items():
            data_dir = self.hosp_dir if category == "hosp" else self.icu_dir
            for table_name in table_names:
                table_path = data_dir / f"{table_name}.parquet"
                if not table_path.exists():
                    return False
        
        return True
    
    def get_table(self, table_name: str) -> pl.LazyFrame:
        """Get a specific table by name."""
        if table_name not in self.tables:
            raise KeyError(f"Table '{table_name}' not found. Available tables: {list(self.tables.keys())}")
        return self.tables[table_name]
    
    def list_tables(self) -> List[str]:
        """List all loaded table names."""
        return list(self.tables.keys())
    
    def get_data_info(self) -> Dict:
        """Get information about the loaded data."""
        return {
            "data_dir": str(self.data_dir),
            "hosp_dir": str(self.hosp_dir),
            "icu_dir": str(self.icu_dir),
            "tables_loaded": self.list_tables(),
            "required_tables": REQUIRED_TABLES
        }

if __name__ == "__main__":
    # Test usage
    pass
