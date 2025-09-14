#!/usr/bin/env python3
"""
Streaming CSV -> Parquet converter (PyArrow).
- Processes only .csv files
- Overwrites existing .parquet files
- Streams record batches directly to Parquet row groups (low memory)
"""

import logging
from pathlib import Path
import pyarrow.csv as pv
import pyarrow.parquet as pq

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SOURCE_ROOT = Path("/gpfs/data/bbj-lab/data/physionet.org/files/mimiciv_parquet")  # adjust if needed
BLOCK_MB = 64  # reduce to 16–32 for tighter RAM

def convert_all():
    if not SOURCE_ROOT.exists():
        logger.error(f"Source directory does not exist: {SOURCE_ROOT}")
        return

    for sub in ("icu", "hosp"):
        d = SOURCE_ROOT / sub
        if d.exists():
            logger.info(f"Processing directory: {d}")
            process_directory(d)

    logger.info("All conversions completed.")

def process_directory(directory: Path):
    csv_files = sorted(directory.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {directory}")

    for csv_path in csv_files:
        parquet_path = csv_path.with_suffix(".parquet")
        logger.info(f"Converting {csv_path.name} -> {parquet_path.name}")

        # Overwrite behavior
        if parquet_path.exists():
            parquet_path.unlink()

        try:
            convert_csv_streaming(csv_path, parquet_path, block_mb=BLOCK_MB)
            logger.info(f"  ✓ Wrote {parquet_path.name}")
        except Exception as e:
            logger.error(f"  ✗ Error converting {csv_path.name}: {e}")

def convert_csv_streaming(csv_path: Path, parquet_path: Path, block_mb: int = 64, column_types: dict | None = None):
    """
    Stream a CSV to a single Parquet file using PyArrow.
    - Processes one RecordBatch at a time
    - Writes each batch immediately as a Parquet row group
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    read_opts = pv.ReadOptions(
        block_size=block_mb * 1024 * 1024,
        use_threads=True,
    )
    convert_opts = pv.ConvertOptions(column_types=column_types) if column_types else pv.ConvertOptions()

    stream = pv.open_csv(
        str(csv_path),
        read_options=read_opts,
        convert_options=convert_opts,
    )

    writer = None
    batches = 0
    try:
        for batch in stream:  # yields pyarrow.RecordBatch
            if writer is None:
                writer = pq.ParquetWriter(
                    where=str(parquet_path),
                    schema=batch.schema,
                    compression="zstd",     # or "snappy" for speed
                    use_dictionary=True,
                )
            writer.write_batch(batch)      # flush batch -> row group on disk
            batches += 1
            if batches % 50 == 0:
                logger.info(f"    wrote {batches} row groups for {csv_path.name}")
    finally:
        if writer is not None:
            writer.close()

def main():
    logger.info("Starting CSV -> Parquet streaming conversion...")
    convert_all()

if __name__ == "__main__":
    main()
