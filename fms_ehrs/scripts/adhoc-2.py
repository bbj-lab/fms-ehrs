#!/usr/bin/env python3

"""
grab the MEDS version of MIMIC
"""

import pathlib

import polars as pl

hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/")

for version in ["mimic-meds-ihlee", "mimic-meds-ed-ihlee"]:
    pl.scan_parquet(
        [hm / version / "train/*.parquet", hm / version / "test/*.parquet"]
    ).sink_parquet(hm / "data-raw" / version / "meds.parquet")

for version, designator in {
    "mimic-meds-ihlee": "raw-meds",
    "mimic-meds-ed-ihlee": "raw-meds-ed",
}.items():
    df = pl.scan_parquet(hm / "data-raw" / version / "meds.parquet")
    df.join(
        df.select(pl.col("subject_id").unique())
        .collect()
        .sample(fraction=0.01, with_replacement=False, seed=42)
        .lazy(),
        on="subject_id",
        validate="m:1",
        how="inner",
    ).sink_parquet(hm / "development-sample-21" / designator / "dev/meds.parquet")


for designator in ["raw-meds", "raw-meds-ed"]:
    print(designator)
    with pl.Config(tbl_rows=-1):
        print(
            pl.read_parquet(
                hm / "development-sample-21" / designator / "dev" / "meds.parquet"
            )
            .select(pl.col("code").str.split("//").list[0])
            .to_series()
            .value_counts()
            .sort("count", descending=True)
        )


df = pl.read_csv(hm / "mimiciv-3.1" / "hosp/admissions.csv.gz").join(
    pl.read_csv(hm / "mimiciv-3.1" / "hosp/patients.csv.gz"),
    on="subject_id",
    validate="m:1",
)
print(df.filter(pl.col("hospitalization_id") == 26886976))

df_asmt = pl.read_parquet(hm / "data-raw/mimic-2.1.0/clif_patient_assessments.parquet")
with pl.Config(tbl_rows=-1):
    print(
        df_asmt.filter(pl.col("hospitalization_id") == "26886976").sort("recorded_dttm")
    )


df_asmt_train = pl.read_parquet(
    hm / "data-mimic/W21/train/clif_patient_assessments.parquet"
)
rass = df_asmt_train.filter(
    (pl.col("assessment_category") == "RASS")
    & pl.col("numerical_value").is_in([-3])
).rename({"recorded_dttm": "rass_dttm"})

cam = df_asmt_train.filter(
    pl.col("assessment_category").str.starts_with("cam_")
).rename({"recorded_dttm": "cam_dttm"})

cam_after_rass = (
    rass.join(cam, on="hospitalization_id")
    .filter(pl.col("rass_dttm") <= pl.col("cam_dttm"))
    .filter(pl.col("rass_dttm") + pl.duration(minutes=15) >= pl.col("cam_dttm"))
    .select(pl.col("rass_dttm").unique())
)

print("{:.2f}".format(len(cam_after_rass) / len(rass)))


# with pl.Config(tbl_rows=-1):
#     print(rass.group_by("numerical_value").len().sort("numerical_value"))
