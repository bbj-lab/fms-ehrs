#!/usr/bin/env python3

"""
Data layer for loading EHR data from different sources.
"""

import os
import pathlib
import typing
from typing import Optional

import polars as pl

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike

# Default data paths
MIMIC_PARQUET_PATH = "/gpfs/data/bbj-lab/data/physionet.org/files/mimiciv_parquet"


class MimicDataLoader:
    """
    Data loader for MIMIC-IV data from parquet files
    """
    
    def __init__(
        self, 
        data_dir: Pathlike = None, 
    ):
        # Initialize data source
        self.data_source = "parquet"
        self.tables: dict[str, pl.LazyFrame] = {}
        
        # Use default path if none provided
        if data_dir is None:
            data_dir = MIMIC_PARQUET_PATH
        
        # Initialize data directory
        self.data_dir = pathlib.Path(data_dir).expanduser()
        
        # Set up directory paths
        self.hosp_dir = self.data_dir / "hosp"
        self.icu_dir = self.data_dir / "icu"
    
    def load_tables(self) -> None:
        """Load MIMIC-IV tables from parquet files"""
        if not self.validate_data_structure():
            raise ValueError(f"Invalid MIMIC-IV parquet data structure in {self.data_dir}")
        
        # Load hospital tables
        self.load_hospital_tables()
        
        # Load ICU tables
        self.load_icu_tables()
    
    def get_required_tables(self) -> dict[str, list[str]]:
        """Return dictionary of required MIMIC-IV table names by category"""
        return {
            "hospital": [
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
    
    def validate_data_structure(self) -> bool:
        """Validate MIMIC-IV parquet directory structure"""
        # Check if base directory exists
        if not self.data_dir.exists():
            return False
        
        # Check if hosp and icu directories exist
        if not self.hosp_dir.exists() or not self.icu_dir.exists():
            return False
        
        # Get required tables by category and check if their files exist
        required_tables = self.get_required_tables()
        for category, table_names in required_tables.items():
            for table_name in table_names:
                table_path = self.get_table_path(table_name)
                if not table_path.exists():
                    return False
        
        return True
    
    def load_hospital_tables(self) -> None:
        """Load tables from the hospital directory"""
        if not self.hosp_dir.exists():
            raise FileNotFoundError(f"Hospital directory not found: {self.hosp_dir}")
        
        # Get hospital tables from required tables
        required_tables = self.get_required_tables()
        hospital_tables = required_tables.get("hospital", [])
        
        # Load each hospital table with data type fixes
        for table_name in hospital_tables:
            table_path = self.get_table_path(table_name)
            if table_path.exists():
                df = pl.scan_parquet(table_path)
                
                # Fix data type mismatches for joins
                if table_name == "labevents":
                    # Ensure itemid is string to match d_labitems
                    df = df.with_columns(pl.col("itemid").cast(pl.Utf8))
                elif table_name == "d_labitems":
                    # Ensure itemid is string to match labevents
                    df = df.with_columns(pl.col("itemid").cast(pl.Utf8))
                
                self.tables[table_name] = df
                print(f"✓ Loaded table: {table_name}")
    
    def load_icu_tables(self) -> None:
        """Load tables from the ICU directory"""
        if not self.icu_dir.exists():
            raise FileNotFoundError(f"ICU directory not found: {self.icu_dir}")
        
        # Get ICU tables from required tables
        required_tables = self.get_required_tables()
        icu_tables = required_tables.get("icu", [])
        
        # Load each ICU table with data type fixes
        for table_name in icu_tables:
            table_path = self.get_table_path(table_name)
            if table_path.exists():
                df = pl.scan_parquet(table_path)
                
                # Fix data type mismatches for joins
                if table_name == "chartevents":
                    # Ensure itemid is string to match d_items
                    df = df.with_columns(pl.col("itemid").cast(pl.Utf8))
                elif table_name == "d_items":
                    # Ensure itemid is string to match chartevents
                    df = df.with_columns(pl.col("itemid").cast(pl.Utf8))
                
                self.tables[table_name] = df
                print(f"✓ Loaded table: {table_name}")
    
    def get_table_path(self, table_name: str) -> pathlib.Path:
        """Get the file path for a specific table"""
        # Get required tables by category
        required_tables = self.get_required_tables()
        
        # Check if table is in hospital category
        if table_name in required_tables.get("hospital", []):
            return self.hosp_dir / f"{table_name}.parquet"
        
        # Check if table is in ICU category
        elif table_name in required_tables.get("icu", []):
            return self.icu_dir / f"{table_name}.parquet"
        
        else:
            raise ValueError(f"Unknown table name: {table_name}")
    
    def get_table_info(self, table_name: str) -> dict:
        """Get information about a specific table"""
        try:
            table_path = self.get_table_path(table_name)
            if not table_path.exists():
                return {
                    "table_name": table_name,
                    "error": "File not found",
                    "source": "parquet"
                }
            
            # Get file size
            file_size = table_path.stat().st_size
            
            # Get schema info
            df = pl.scan_parquet(table_path)
            schema = df.schema
            
            return {
                "table_name": table_name,
                "file_size_bytes": file_size,
                "file_size_mb": file_size / (1024**2),
                "columns": list(schema.keys()),
                "dtypes": {k: str(v) for k, v in schema.items()},
                "source": "parquet"
            }
            
        except Exception as e:
            return {
                "table_name": table_name,
                "error": str(e),
                "source": "parquet"
            }
    
    def get_all_required_tables(self) -> list[str]:
        """Get all required table names as a flat list (for backward compatibility)"""
        required_tables = self.get_required_tables()
        all_tables = []
        for table_names in required_tables.values():
            all_tables.extend(table_names)
        return all_tables
    
    def get_data_info(self) -> dict:
        """Get information about the loaded data"""
        info = {
            "data_source": "parquet",
            "data_dir": str(self.data_dir),
            "hosp_dir": str(self.hosp_dir),
            "icu_dir": str(self.icu_dir),
            "tables_loaded": self.list_tables(),
            "table_count": self.get_table_count(),
            "required_tables_by_category": self.get_required_tables(),
            "all_required_tables": self.get_all_required_tables()
        }
        return info
    
    def get_table(self, table_name: str) -> pl.LazyFrame:
        """Get a specific table by name"""
        if table_name not in self.tables:
            raise KeyError(f"Table '{table_name}' not found. Available tables: {list(self.tables.keys())}")
        return self.tables[table_name]
    
    def list_tables(self) -> list[str]:
        """List all loaded table names"""
        return list(self.tables.keys())
    
    def is_table_loaded(self, table_name: str) -> bool:
        """Check if a specific table is loaded"""
        return table_name in self.tables
    
    def get_table_count(self) -> int:
        """Get the number of loaded tables"""
        return len(self.tables)
    
    def clear_tables(self) -> None:
        """Clear all loaded tables"""
        self.tables.clear()
    
    def reload_tables(self) -> None:
        """Reload all tables"""
        self.clear_tables()
        self.load_tables()


class ClifDataLoader:
    """
    Data loader for CLIF parquet files (placeholder)
    """
    
    def __init__(self, data_dir: Pathlike):
        self.data_dir = pathlib.Path(data_dir).expanduser()
        self.tables: dict[str, pl.LazyFrame] = {}
    
    def load_tables(self) -> None:
        """Load CLIF tables from parquet files"""
        pass  # Implementation to be added
    
    def get_required_tables(self) -> dict[str, list[str]]:
        """Return dictionary of required CLIF table names by category"""
        return {
            "core": [
                "patient",
                "hospitalization"
            ],
            "events": [
                "adt",
                "labs", 
                "vitals",
                "medication",
                "assessments",
                "respiratory",
                "position"
            ]
        }
    
    def validate_data_structure(self) -> bool:
        """Validate CLIF directory structure"""
        pass  # Implementation to be added
    
    def get_table_name_from_file(self, file_path: pathlib.Path) -> str:
        """Extract table name from CLIF parquet filename"""
        pass  # Implementation to be added


class DatabaseDataLoader:
    """
    Data loader for database connections (placeholder)
    """
    
    def __init__(self, connection_string: str, schema: str = None):
        self.connection_string = connection_string
        self.schema = schema
        self.tables: dict[str, pl.LazyFrame] = {}
    
    def load_tables(self) -> None:
        """Load tables from database connection"""
        pass  # Implementation to be added
    
    def get_required_tables(self) -> dict[str, list[str]]:
        """Return dictionary of required database table names by category"""
        return {}  # To be defined based on database schema
    
    def validate_data_structure(self) -> bool:
        """Validate database connection and schema"""
        pass  # Implementation to be added
    
    def execute_query(self, query: str) -> pl.LazyFrame:
        """Execute SQL query and return as LazyFrame"""
        pass  # Implementation to be added


if __name__ == "__main__":
    # Example usage
    print("=== MIMIC-IV Data Loader Examples ===\n")
    
    # Example 1: Default parquet mode
    print("1. Default Parquet Mode Example:")
    try:
        mimic_loader = MimicDataLoader()
        mimic_loader.load_tables()
        print(f"   Data source: {mimic_loader.data_source}")
        print(f"   Loaded tables: {mimic_loader.list_tables()}")
        print(f"   Data info: {mimic_loader.get_data_info()}")
    except Exception as e:
        print(f"   Parquet mode failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Custom parquet path
    print("2. Custom Parquet Path Example:")
    print("   # mimic_loader_custom = MimicDataLoader('/custom/path/to/mimic_parquet')")
    print("   # mimic_loader_custom.load_tables()")
    print("   # print(f'Loaded tables: {mimic_loader_custom.list_tables()}')")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: CLIF (placeholder)
    # print("3. CLIF Data Loader (placeholder):")
    # print("   # clif_loader = ClifDataLoader('/path/to/clif/data')")
    # print("   # clif_loader.load_tables()")
    # print("   # print(f'Loaded tables: {clif_loader.list_tables()}')")
    
    # print("\n" + "="*50 + "\n")
    
    # # Example 4: Table information
    # print("4. Table Information Example:")
    # print("   # mimic_loader = MimicDataLoader()")
    # print("   # mimic_loader.load_tables()")
    # print("   # info = mimic_loader.get_table_info('patients')")
    # print("   # print(f'Table info: {info}')")