#!/usr/bin/env python3

"""
Extract development sample consisting of data associated to the first 10k
patients with records
"""

import pathlib

import polars as pl

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")
data_dir = hm.joinpath("CLIF-MIMIC", "output", "rclif-2.1")
dev_dir = hm.joinpath("clif-development-sample")

# grab patient ids
dev_p_ids = (
    pl.scan_parquet(data_dir.joinpath("clif_hospitalization.parquet"))
    .group_by("patient_id")
    .agg(pl.col("admission_dttm").min().alias("first_admission"))
    .sort("first_admission")
    .head(10_000)
    .select("patient_id")
)

# look up hospitalization ids for selected patients
dev_h_ids = (
    pl.scan_parquet(data_dir.joinpath("clif_hospitalization.parquet"))
    .join(dev_p_ids, on="patient_id")
    .select("hospitalization_id")
)

# generate sub-tables
pl.scan_parquet(data_dir.joinpath("clif_patient.parquet")).join(
    dev_p_ids, on="patient_id"
).sink_parquet(dev_dir.joinpath("clif_patient.parquet"))

pl.scan_parquet(data_dir.joinpath("clif_hospitalization.parquet")).join(
    dev_h_ids, on="hospitalization_id"
).sink_parquet(dev_dir.joinpath("clif_hospitalization.parquet"))

for t in data_dir.glob("*.parquet"):
    if t.stem.split("_", 1)[1] not in ("hospitalization", "patient"):
        pl.scan_parquet(t).join(dev_h_ids, on="hospitalization_id").sink_parquet(
            dev_dir.joinpath(t.name)
        )

# summarize results
print("-" * 42)
for t in dev_dir.glob("*.parquet"):
    print(t.stem)
    pl.read_parquet(t).glimpse()
    print("-" * 42)
