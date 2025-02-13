#!/usr/bin/env python3

"""
partition patients by appearance in the dataset into train-validation-test sets
at a 70%-10%-20% split
"""

import itertools
import pathlib

import polars as pl

version_name = "raw"
hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")
data_dir = hm.joinpath("CLIF-MIMIC", "output", "rclif-2.1")
splits = ("train", "val", "test")

# partition patient ids
patient_ids = (
    pl.scan_parquet(data_dir.joinpath("clif_hospitalization.parquet"))
    .filter(pl.col("age_at_admission") >= 18)
    .group_by("patient_id")
    .agg(pl.col("admission_dttm").min().alias("first_admission"))
    .sort("first_admission")
    .select("patient_id")
    .collect()
)

n_total = patient_ids.n_unique()
n_train = int(0.7 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - (n_train + n_val)

p_ids = dict()
p_ids["train"] = patient_ids.head(n_train)
p_ids["val"] = patient_ids.slice(n_train, n_val)
p_ids["test"] = patient_ids.tail(n_test)

for s0, s1 in itertools.combinations(splits, 2):
    assert p_ids[s0].join(p_ids[s1], on="patient_id").n_unique() == 0

assert sum(list(map(lambda x: x.n_unique(), p_ids.values()))) == n_total

# partition hospitalization ids according to the patient split
hospitalization_ids = (
    pl.scan_parquet(data_dir.joinpath("clif_hospitalization.parquet"))
    .select("patient_id", "hospitalization_id")
    .unique()
    .collect()
)

h_ids = dict()
for s in splits:
    h_ids[s] = hospitalization_ids.join(p_ids[s], on="patient_id").select(
        "hospitalization_id"
    )

for s0, s1 in itertools.combinations(splits, 2):
    assert h_ids[s0].join(h_ids[s1], on="hospitalization_id").n_unique() == 0

assert (
    sum(list(map(lambda x: x.n_unique(), h_ids.values())))
    == hospitalization_ids.n_unique()
)

# make directories
dirs = dict()
for s in splits:
    dirs[s] = hm.joinpath("clif-data", version_name, s)
    dirs[s].mkdir(exist_ok=True, parents=True)

# generate sub-tables
for s in splits:

    pl.scan_parquet(data_dir.joinpath("clif_patient.parquet")).join(
        p_ids[s].lazy(), on="patient_id"
    ).sink_parquet(dirs[s].joinpath("clif_patient.parquet"))

    pl.scan_parquet(data_dir.joinpath("clif_hospitalization.parquet")).join(
        h_ids[s].lazy(), on="hospitalization_id"
    ).sink_parquet(dirs[s].joinpath("clif_hospitalization.parquet"))

    for t in data_dir.glob("*.parquet"):
        if t.stem.split("_", 1)[1] not in ("hospitalization", "patient"):
            pl.scan_parquet(t).join(
                h_ids[s].lazy(), on="hospitalization_id"
            ).sink_parquet(dirs[s].joinpath(t.name))

# display results
print("-" * 42)
for s in splits:
    print(s.upper().center(42, "="))
    for t in dirs[s].glob("*.parquet"):
        print(t.stem)
        pl.read_parquet(t).glimpse()
        print("-" * 42)
