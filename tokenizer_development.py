#!/usr/bin/env python3

import gzip
import os
import pathlib
import pickle
import polars as pl
import numpy as np

if os.uname().nodename.startswith("cri"):
    hm = pathlib.Path(
        "/gpfs/data/bbj-lab/users/burkh4rt/clif-development-sample"
    )
else:
    # change following line to develop locally
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

    def save(self, filepath: pathlib.PurePath | str):
        with gzip.open(pathlib.Path(filepath).expanduser(), "wb") as f:
            pickle.dump(
                {
                    "lookup": self.lookup,
                    "reverse": self.reverse,
                    "aux": {k: list(v) for k, v in self.aux.items()},
                },
                f,
            )

    def load(self, filepath: pathlib.PurePath | str):
        with gzip.open(pathlib.Path(filepath).expanduser(), mode="rb") as f:
            for k, v in pickle.load(f).items():
                setattr(self, k, v)


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


tbl = dict()

""" Patients
"""

tbl["patient"] = (
    pl.scan_parquet(hm.joinpath("patient.parquet"))
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

tbl["hospitalization"] = (
    pl.scan_parquet(hm.joinpath("hospitalization.parquet"))
    .group_by("hospitalization_id")
    .agg(
        pl.col("patient_id").first(),
        pl.col("admission_dttm").first().cast(pl.Datetime(time_unit="ms")),
        pl.col("discharge_dttm").first().cast(pl.Datetime(time_unit="ms")),
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
v = tbl["hospitalization"].select("age_at_admission").to_numpy().ravel()
if not vocab.is_aux(c):
    vocab.set_aux(c, np.nanquantile(v, np.arange(0.1, 1.0, 0.1)))
tbl["hospitalization"] = (
    tbl["hospitalization"]
    .with_columns(age_at_admission=np.digitize(v, bins=vocab.get_aux(c)))
    .with_columns(
        admission_tokens=pl.concat_list(
            "age_at_admission", "admission_type_name"
        ),
    )
    .drop("age_at_admission", "admission_type_name")
)


""" Adt
"""

tbl["adt"] = (
    pl.scan_parquet(hm.joinpath("adt.parquet"))
    .rename(
        {
            "in_dttm": "event_time",
            "out_dttm": "event_end",
            "location_category": "category",
        }
    )
    .cast(
        {
            "event_time": pl.Datetime(time_unit="ms"),
            "event_end": pl.Datetime(time_unit="ms"),
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
    .cast({"times": pl.List(pl.Datetime(time_unit="ms"))})
    .collect()
)


""" Labs
"""

tbl["labs"] = (
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
    .cast(
        {
            "event_start": pl.Datetime(time_unit="ms"),
            "event_time": pl.Datetime(time_unit="ms"),
        }
    )
    .select(
        "hospitalization_id", "event_start", "event_time", "category", "value"
    )
    .collect()
)
tbl["labs"] = process_cat_val_frame(tbl["labs"], label="LAB")

""" Vitals
"""

tbl["vitals"] = (
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
    .cast(
        {
            "event_time": pl.Datetime(time_unit="ms"),
        }
    )
    .select("hospitalization_id", "event_time", "category", "value")
    .collect()
)
tbl["vitals"] = process_cat_val_frame(tbl["vitals"], label="VTL")

""" Medication Admin Continuous
"""

tbl["medication"] = (
    pl.scan_parquet(hm.joinpath("medication_admin_continuous.parquet"))
    .rename(
        {
            "admin_dttm": "event_time",
            "med_category": "category",
            "med_dose": "value",
        }
    )
    .cast(
        {
            "event_time": pl.Datetime(time_unit="ms"),
        }
    )
    .select("hospitalization_id", "event_time", "category", "value")
    .collect()
)
tbl["medication"] = process_cat_val_frame(tbl["medication"], label="MED")

""" Patient Assessments
"""

tbl["assessments"] = (
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
    .cast(
        {
            "event_time": pl.Datetime(time_unit="ms"),
        }
    )
    .select("hospitalization_id", "event_time", "category", "value")
    .collect()
)
tbl["assessments"] = process_cat_val_frame(tbl["assessments"], label="ASM")


""" Respiratory Support
"""

tbl["respiratory"] = (
    pl.scan_parquet(
        hm.joinpath("respiratory_support.parquet"),
    )
    .rename(
        {
            "recorded_dttm": "event_time",
        }
    )
    .cast(
        {
            "event_time": pl.Datetime(time_unit="ms"),
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

## prepend patient-level tokens to each admission event
admission_tokens = (
    tbl["patient"]
    .join(tbl["hospitalization"], on="patient_id", validate="1:m")
    .cast(
        {
            "event_start": pl.Datetime(time_unit="ms"),
        }
    )
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
    tbl["hospitalization"]
    .rename({"event_end": "event_time"})
    .cast(
        {
            "event_time": pl.Datetime(time_unit="ms"),
        }
    )
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
    tbl[k] for k in tbl.keys() if k not in ("patient", "hospitalization")
)

"""
doing both aggregations at once doesn't seem to work
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
