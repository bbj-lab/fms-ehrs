import polars as pl

# Load the CLIF patient assessments table
# df = pl.read_parquet("/gpfs/data/bbj-lab/users/burkh4rt/development-sample/raw/clif_patient_assessments.parquet")
df = pl.read_parquet("/home/chend5/fms-ehrs/fms_ehrs/framework/mimiciv_timelines.parquet")

# Show basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nData types:")
print(df.dtypes)

# Show first few rows
print("\nFirst 5 rows:")
print(df.head())

# Show unique values in key columns
# print("\nUnique assessment categories:")
# print(df["assessment_category"].value_counts().head(10))

# print("\nUnique numerical values (sample):")
# print(df["numerical_value"].drop_nulls().head(10))

# print("\nUnique categorical values (sample):")
# print(df["categorical_value"].drop_nulls().head(10))

# # Check time column
# print("\nTime column info:")
# print("Unique times (first 10):", df["recorded_dttm"].head(10))
# print("Time range:", df["recorded_dttm"].min(), "to", df["recorded_dttm"].max())