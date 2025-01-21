#!/usr/bin/env python3

import pathlib
import polars as pl

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")
data_dir = hm.joinpath("CLIF-MIMIC", "rclif")
dev_dir = hm.joinpath("clif-development-sample")

""" Extract development sample
"""

development_patient_ids = (
    pl.scan_parquet(data_dir.joinpath("clif_hospitalization.parquet"))
    .group_by("patient_id")
    .agg(pl.col("admission_dttm").min().alias("first_admission"))
    .sort("first_admission")
    .head(10_000)
    .select("patient_id")
)

development_hospitalization_ids = (
    pl.scan_parquet(data_dir.joinpath("clif_hospitalization.parquet"))
    .join(development_patient_ids, on="patient_id")
    .select("hospitalization_id")
)

pl.read_parquet(data_dir.joinpath("clif_patient.parquet")).join(
    development_patient_ids.collect(), on="patient_id"
).write_parquet(dev_dir.joinpath("patient.parquet"))

pl.read_parquet(data_dir.joinpath("clif_hospitalization.parquet")).join(
    development_hospitalization_ids.collect(), on="hospitalization_id"
).write_parquet(dev_dir.joinpath("hospitalization.parquet"))

for t in data_dir.glob("*.parquet"):
    if t.stem.split("_", 1)[1] not in ("hospitalization", "patient"):
        pl.scan_parquet(t).join(
            development_hospitalization_ids, on="hospitalization_id"
        ).sink_parquet(dev_dir.joinpath(t.stem.split("_", 1)[1] + ".parquet"))

print("-" * 42)
for t in dev_dir.glob("*.parquet"):
    print(t.stem)
    pl.read_parquet(t).glimpse()
    print("-" * 42)
