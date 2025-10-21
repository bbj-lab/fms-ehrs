import polars as pl

# Path to your Parquet file
file_path = "/home/chend5/fms-ehrs/fms_ehrs/framework/mimiciv_timelines.parquet"

# Read the Parquet file
df = pl.read_parquet(file_path)

# Show the first 20 rows
print(df.head(20))
