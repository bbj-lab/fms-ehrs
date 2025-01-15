#!/usr/bin/env python3

import os
import pathlib
import polars as pl
import numpy as np

"""
Grab the data from `/gpfs/data/bbj-lab/users/burkh4rt/clif-development-sample`
and change the `hm` path that follows as necessary:
"""
if os.uname().nodename.startswith("cri"):
    hm = pathlib.Path(
        "/gpfs/data/bbj-lab/users/burkh4rt/clif-development-sample"
    )
else:
    hm = pathlib.Path("~/Documents/chicago/CLIF/clif-development-sample")


class Vocabulary:
    """
    maintains a dictionary `lookup` mapping words -> tokens,
    a dictionary `reverse` inverting the lookup, and a dictionary
    `aux` mapping words -> auxiliary info
    """

    def __init__(self, words: tuple = ()):
        assert len(set(words)) == len(words)
        self.lookup = {v: i for i, v in enumerate(words)}
        self.reverse = dict(enumerate(words))
        self.aux = {}

    def __call__(self, word: str):
        try:
            return self.lookup[word]
        except KeyError:
            self.lookup[word], self.reverse[n] = (n := len(self.lookup)), word
            return n

    def set_aux(self, word: str, aux_data):
        self.aux[word] = aux_data

    def is_aux(self, word: str):
        return word in self.aux

    def get_aux(self, word: str):
        return self.aux[word]


vocab = Vocabulary(tuple(map(lambda i: f"Q{i}", range(10))))


def process_single_category(x, label):
    v = x.select("value").to_numpy().ravel()
    c = x.select("category").row(0)[0]
    if not vocab.is_aux(f"{label}_{c}"):
        vocab.set_aux(
            f"{label}_{c}", np.nanquantile(v, np.arange(0.1, 1.0, 0.1))
        )
    return (
        x.with_columns(
            token=vocab(f"{label}_{c}"),
            token_quantile=np.digitize(v, bins=vocab.get_aux(f"{label}_{c}")),
        )
        .with_columns(
            tokens=pl.concat_list("token", "token_quantile"),
            times=pl.concat_list("event_time", "event_time"),
        )
        .select("hospitalization_id", "event_time", "tokens", "times")
    )


def process_cat_val_frame(df, label):
    """handle tables that"""
    return pl.concat(
        process_single_category(x, label) for x in df.partition_by("category")
    )


""" Patients
"""

df_patients = (
    pl.scan_parquet(hm.joinpath("patients.parquet"))
    .select(
        "patient_id", "race_category", "ethnicity_category", "sex_category"
    )
    .group_by("patient_id")
    .agg(
        pl.col("race_category").first(),
        pl.col("ethnicity_category").first(),
        pl.col("sex_category").first(),
    )
    .with_columns(
        pl.col("race_category").map_elements(
            vocab, return_dtype=pl.Int64, skip_nulls=False
        ),
        pl.col("ethnicity_category").map_elements(
            vocab, return_dtype=pl.Int64, skip_nulls=False
        ),
        pl.col("sex_category").map_elements(
            vocab, return_dtype=pl.Int64, skip_nulls=False
        ),
    )
    .with_columns(
        tokens=pl.concat_list(
            "race_category", "ethnicity_category", "sex_category"
        ),
    )
    .select("patient_id", "tokens")
    .collect()
)

""" Hospitalization
"""

df_hospitalization = (
    pl.scan_parquet(hm.joinpath("hospitalization.parquet"))
    .group_by("hospitalization_id")
    .agg(
        pl.col("patient_id").first(),
        pl.col("admission_dttm").first(),
        pl.col("discharge_dttm").first(),
        pl.col("age_at_admission").first(),
        pl.col("admission_type_name").first(),
        pl.col("discharge_category").first(),
    )
    .rename(
        {
            "admission_dttm": "event_start",
            "discharge_dttm": "event_end",
        }
    )
    .with_columns(
        pl.col("admission_type_name").map_elements(
            vocab, return_dtype=pl.Int64, skip_nulls=False
        ),
        pl.col("discharge_category").map_elements(
            vocab, return_dtype=pl.Int64, skip_nulls=False
        ),
    )
    .select(
        "patient_id",
        "hospitalization_id",
        "event_start",
        "event_end",
        "age_at_admission",
        "admission_type_name",
        "discharge_category",
    )
    .collect()
)

# tokenize age_at_admission here
c = "age_at_admission"
v = df_hospitalization.select("age_at_admission").to_numpy().ravel()
if not vocab.is_aux(c):
    vocab.set_aux(c, np.nanquantile(v, np.arange(0.1, 1.0, 0.1)))
df_hospitalization = (
    df_hospitalization.with_columns(
        age_at_admission=np.digitize(v, bins=vocab.get_aux(c))
    )
    .with_columns(
        admission_tokens=pl.concat_list(
            "age_at_admission", "admission_type_name"
        ),
    )
    .drop("age_at_admission", "admission_type_name")
)


""" Adt
"""

df_adt = (
    pl.scan_parquet(hm.joinpath("adt.parquet"))
    .rename(
        {
            "in_dttm": "event_time",
            "out_dttm": "event_end",
            "location_category": "category",
        }
    )
    .with_columns(
        tokens=pl.col("category").map_elements(
            lambda x: [vocab(x)],
            return_dtype=pl.List(pl.Int64),
            skip_nulls=False,
        ),
        times=pl.col("event_time").map_elements(
            lambda x: [x],
            return_dtype=pl.List(pl.Datetime),
            skip_nulls=False,
        ),
    )
    .select("hospitalization_id", "event_time", "tokens", "times")
    .cast({"times": pl.List(pl.Datetime(time_unit="ns"))})
    .collect()
)


""" Labs
"""

df_labs = (
    pl.scan_parquet(
        hm.joinpath("labs.parquet"),
    )
    .rename(
        {
            "lab_collect_dttm": "event_start",
            "lab_result_dttm": "event_time",
            "lab_category": "category",
            "lab_value_numeric": "value",
        }
    )
    .select(
        "hospitalization_id", "event_start", "event_time", "category", "value"
    )
    .collect()
)
df_labs_procd = process_cat_val_frame(df_labs, label="LAB")

""" Vitals
"""

df_vitals = (
    pl.scan_parquet(
        hm.joinpath("vitals.parquet"),
    )
    .rename(
        {
            "recorded_dttm": "event_time",
            "vital_category": "category",
            "vital_value": "value",
        }
    )
    .select("hospitalization_id", "event_time", "category", "value")
    .collect()
)
df_vitals_procd = process_cat_val_frame(df_vitals, label="VTL")

""" Medication Admin Continuous
"""

df_medication = (
    pl.scan_parquet(hm.joinpath("medication_admin_continuous.parquet"))
    .rename(
        {
            "admin_dttm": "event_time",
            "med_category": "category",
            "med_dose": "value",
        }
    )
    .select("hospitalization_id", "event_time", "category", "value")
    .collect()
)
df_medication_procd = process_cat_val_frame(df_medication, label="MED")

""" Patient Assessments
"""

df_assessments = (
    pl.scan_parquet(
        hm.joinpath("patient_assessments.parquet"),
    )
    .rename(
        {
            "recorded_dttm": "event_time",
            "assessment_category": "category",
            "numerical_value": "value",
        }
    )
    .select("hospitalization_id", "event_time", "category", "value")
    .collect()
)
df_assessments_procd = process_cat_val_frame(df_assessments, label="ASM")


""" Respiratory Support
"""

df_respiratory = (
    pl.scan_parquet(
        hm.joinpath("respiratory_support.parquet"),
    )
    .rename(
        {
            "recorded_dttm": "event_time",
        }
    )
    .with_columns(
        pl.col("mode_category").map_elements(
            vocab, return_dtype=pl.Int64, skip_nulls=False
        ),
        pl.col("device_category").map_elements(
            vocab, return_dtype=pl.Int64, skip_nulls=False
        ),
    )
    .with_columns(
        tokens=pl.concat_list("mode_category", "device_category"),
        times=pl.concat_list("event_time", "event_time"),
    )
    .select("hospitalization_id", "event_time", "tokens", "times")
    .collect()
)

"""
"""

df_patients.with_columns(
    tokens=pl.concat_list(pl.exclude("patient_id"))
).select("patient_id", "tokens")


## prepend patient-level tokens to each admission event
admission_tokens = (
    df_patients.join(df_hospitalization, on="patient_id", validate="1:m")
    .with_columns(
        adm_tokens=pl.concat_list(
            pl.col("tokens"), pl.col("admission_tokens")
        ),
        adm_times=pl.concat_list(*[pl.col("event_start")] * 5),
    )
    .select(
        "hospitalization_id",
        pl.col("event_start").alias("event_time"),
        "adm_tokens",
        "adm_times",
    )
)

# gather discharge tokens
discharge_tokens = (
    df_hospitalization.rename({"event_end": "event_time"})
    .with_columns(
        dis_tokens=pl.col("discharge_category").map_elements(
            lambda x: [vocab(x)],
            return_dtype=pl.List(pl.Int64),
            skip_nulls=False,
        ),
        dis_times=pl.col("event_time").map_elements(
            lambda x: [x],
            return_dtype=pl.List(pl.Datetime),
            skip_nulls=False,
        ),
    )
    .select("hospitalization_id", "event_time", "dis_tokens", "dis_times")
)

events = pl.concat(
    [
        df_adt,
        df_labs_procd,
        df_vitals_procd,
        df_medication_procd,
        df_assessments_procd,
        df_respiratory,
    ]
)

"""
for some reason, doing both aggregations at once doesn't seem to work
"""

tokens_agg = (
    events.lazy()
    .sort("event_time")
    .group_by("hospitalization_id", maintain_order=True)
    .agg([pl.col("tokens").explode()])
)

times_agg = (
    events.lazy()
    .sort("event_time")
    .group_by("hospitalization_id", maintain_order=True)
    .agg(
        [pl.col("times").explode()],
    )
)

event_tokens = tokens_agg.join(times_agg, on="hospitalization_id")

# combine the admission tokens, event tokens, and discharge tokens
timelines_tokens = (
    admission_tokens.lazy()
    .join(event_tokens, on="hospitalization_id")
    .join(discharge_tokens.lazy(), on="hospitalization_id")
    .with_columns(
        tokens=pl.concat_list("adm_tokens", "tokens", "dis_tokens"),
        times=pl.concat_list("adm_times", "times", "dis_times"),
    )
    .select("hospitalization_id", "tokens", "times")
    .collect()
)
