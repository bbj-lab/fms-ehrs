#!/usr/bin/env python3

import os
import pathlib
import typing

import numpy as np
import polars as pl

from vocabulary import Vocabulary

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str


class ClifTokenizer:
    """
    tokenizes a directory containing a set of parquet files corresponding to
    the CLIF-2.0 standard
    """

    def __init__(
        self,
        *,
        data_dir: Pathlike = pathlib.Path("."),
        vocab_path: Pathlike = None,
    ):
        """
        if no vocabulary is provided, we are in training mode; otherwise, the
        provided vocabulary is frozen
        """
        self.data_dir = pathlib.Path(data_dir).expanduser()
        self.tbl = dict()
        if vocab_path is None:
            self.vocab_path = None
            self.vocab = Vocabulary(tuple(map(lambda i: f"Q{i}", range(10))))
            self.vocab.is_training = True
        else:
            self.vocab_path = pathlib.Path(vocab_path).expanduser()
            self.vocab = Vocabulary().load(self.vocab_path)
            self.vocab.is_training = False

    def load_tables(self):
        """lazy-load all parquet tables from the directory `self.data_dir`"""
        self.tbl = {
            (
                p.stem.split("_")[1] if "assessments" not in p.stem else "assessments"
            ): pl.scan_parquet(p)
            for p in self.data_dir.glob("*.parquet")
        }

    def process_single_category(self, x: Frame, label: str) -> Frame:
        """
        Quantize a sub-table consisting of a single category

        The way our quantization works, if a category takes on only a single
        value, then this value is sent to the Q9 token, because, e.g.
        `np.digitize(1, bins=[1] * 9) == 9`
        and:
        `np.digitize(
        [1, 2],
        bins=np.nanquantile([1, 1, 1, 2, 2, 2, 2], np.arange(0.1, 1.0, 0.1)),
        ) == [3, 9]`
        This is why the Q9 token appears quite a bit more often in our dataset than
        certain other quantile tokens.
        """
        v = x.select("value").to_numpy().ravel()
        c = x.select("category").row(0)[0]
        if not self.vocab.has_aux(f"{label}_{c}"):
            self.vocab.set_aux(
                f"{label}_{c}", np.nanquantile(v, np.arange(0.1, 1.0, 0.1))
            )
        return (
            x.with_columns(
                token=pl.lit(self.vocab(f"{label}_{c}")),
                token_quantile=pl.lit(
                    pl.Series(
                        np.where(
                            np.isfinite(v),
                            np.digitize(v, bins=self.vocab.get_aux(f"{label}_{c}")),
                            self.vocab("nan"),
                        )
                    ),
                ),
            )
            .drop_nulls("token")
            .with_columns(
                tokens=pl.concat_list("token", "token_quantile"),
                times=pl.concat_list("event_time", "event_time"),
            )
        )

    def process_cat_val_frame(self, df: Frame, label: str) -> Frame:
        """handle tables that can mostly be described in terms of categories and
        values"""
        return pl.concat(
            self.process_single_category(x, label)
            for k, x in df.partition_by("category", as_dict=True).items()
        )

    def process_tables(self):

        self.tbl["patient"] = (
            self.tbl["patient"]
            .select("patient_id", "race_category", "ethnicity_category", "sex_category")
            .group_by("patient_id")
            .agg(
                pl.col("race_category").first(),
                pl.col("ethnicity_category").first(),
                pl.col("sex_category").first(),
            )
            .with_columns(
                pl.col("race_category").map_elements(
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
                ),
                pl.col("ethnicity_category").map_elements(
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
                ),
                pl.col("sex_category").map_elements(
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
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

        self.tbl["hospitalization"] = (
            self.tbl["hospitalization"]
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
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
                ),
                pl.col("discharge_category").map_elements(
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
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
        v = self.tbl["hospitalization"].select("age_at_admission").to_numpy().ravel()
        if not self.vocab.has_aux(c):
            self.vocab.set_aux(c, np.nanquantile(v, np.arange(0.1, 1.0, 0.1)))
        self.tbl["hospitalization"] = (
            self.tbl["hospitalization"]
            .with_columns(
                age_at_admission=pl.lit(
                    pl.Series(
                        np.where(
                            np.isfinite(v),
                            np.digitize(v, bins=self.vocab.get_aux(c)),
                            self.vocab("nan"),
                        )
                    ),
                )
            )
            .with_columns(
                admission_tokens=pl.concat_list(
                    "age_at_admission", "admission_type_name"
                ),
            )
            .drop("age_at_admission", "admission_type_name")
        )

        self.tbl["adt"] = (
            self.tbl["adt"]
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
                    lambda x: [self.vocab(x)],
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

        self.tbl["labs"] = (
            self.tbl["labs"]
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
                "hospitalization_id",
                "event_start",
                "event_time",
                "category",
                "value",
            )
            .collect()
        )
        self.tbl["labs"] = self.process_cat_val_frame(self.tbl["labs"], label="LAB")

        self.tbl["vitals"] = (
            self.tbl["vitals"]
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
        self.tbl["vitals"] = self.process_cat_val_frame(self.tbl["vitals"], label="VTL")

        self.tbl["medication"] = (
            self.tbl["medication"]
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
        self.tbl["medication"] = self.process_cat_val_frame(
            self.tbl["medication"], label="MED"
        )

        # seems like there's a column for assessment, and then either a
        # numerical_value OR a categorical_value, depending on the assessment
        self.tbl["assessments"] = (
            self.tbl["assessments"]
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
            .collect()
        )

        # handle categorical assessments separately from numerical assessments
        asmt_num = self.tbl["assessments"].filter(~pl.col("value").is_null())
        asmt_num = self.process_cat_val_frame(asmt_num, label="ASMT").select(
            "hospitalization_id", "event_time", "tokens", "times"
        )

        asmt_cat = (
            self.tbl["assessments"]
            .filter(pl.col("value").is_null())
            .filter(~pl.col("categorical_value").is_null())
            .with_columns(
                pl.col("category").map_elements(
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
                ),
                pl.col("categorical_value").map_elements(
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
                ),
            )
            .with_columns(
                tokens=pl.concat_list("category", "categorical_value"),
                times=pl.concat_list("event_time", "event_time"),
            )
            .select("hospitalization_id", "event_time", "tokens", "times")
        )

        self.tbl["assessments"] = pl.concat((asmt_num, asmt_cat))

        self.tbl["respiratory"] = (
            self.tbl["respiratory"]
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
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
                ),
                pl.col("device_category").map_elements(
                    self.vocab, return_dtype=pl.Int64, skip_nulls=False
                ),
            )
            .with_columns(
                tokens=pl.concat_list("mode_category", "device_category"),
                times=pl.concat_list("event_time", "event_time"),
            )
            .select("hospitalization_id", "event_time", "tokens", "times")
            .collect()
        )

        # include a token for prone position; this is relatively rare
        self.tbl["position"] = (
            self.tbl["position"]
            .collect()
            .filter(pl.col("position_category") == "prone")
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
                tokens=pl.col("position_category").map_elements(
                    lambda x: [self.vocab(x)],
                    return_dtype=pl.List(pl.Int64),
                    skip_nulls=False,
                ),
                times=pl.col("event_time").map_elements(
                    lambda x: [x],
                    return_dtype=pl.List(pl.Datetime),
                    skip_nulls=False,
                ),
            )
            .cast({"times": pl.List(pl.Datetime(time_unit="ms"))})
        )

    def get_admission_frame(self) -> Frame:

        ## prepend patient-level tokens to each admission event
        admission_tokens = (
            self.tbl["patient"]
            .join(self.tbl["hospitalization"], on="patient_id", validate="1:m")
            .cast(
                {
                    "event_start": pl.Datetime(time_unit="ms"),
                }
            )
            .with_columns(
                adm_tokens=pl.concat_list(
                    pl.lit(self.vocab("TL_START")),
                    pl.col("tokens"),
                    pl.col("admission_tokens"),
                ),
                adm_times=pl.concat_list(*[pl.col("event_start")] * 6),
            )
            .select(
                "hospitalization_id",
                pl.col("event_start").alias("event_time"),
                "adm_tokens",
                "adm_times",
            )
        )

        return admission_tokens

    def get_discharge_frame(self) -> Frame:
        # gather discharge tokens
        discharge_tokens = (
            self.tbl["hospitalization"]
            .rename({"event_end": "event_time"})
            .cast(
                {
                    "event_time": pl.Datetime(time_unit="ms"),
                }
            )
            .with_columns(
                dis_tokens=pl.concat_list(
                    "discharge_category", pl.lit(self.vocab("TL_END"))
                ),
                dis_times=pl.concat_list(*[pl.col("event_time")] * 2),
            )
            .cast({"dis_times": pl.List(pl.Datetime(time_unit="ms"))})
            .select("hospitalization_id", "event_time", "dis_tokens", "dis_times")
        )

        return discharge_tokens

    def get_events_frame(self) -> Frame:
        events = pl.concat(
            self.tbl[k].select("hospitalization_id", "event_time", "tokens", "times")
            for k in self.tbl.keys()
            if k not in ("patient", "hospitalization")
        )

        # doing both aggregations at once doesn't seem to work; so we do them
        # separately, lazily, and then stitch them together

        tokens_agg = (
            events.lazy()
            # order concurrent events by vocabulary, which itself was formed with
            # contiguous categories
            .sort("event_time", pl.col("tokens").list.first())
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
        return event_tokens

    def get_tokens_timelines(self) -> tuple[Frame, dict] | Frame:
        self.load_tables()
        self.process_tables()

        # combine the admission tokens, event tokens, and discharge tokens
        return (
            self.get_admission_frame()
            .lazy()
            .join(self.get_events_frame(), on="hospitalization_id")
            .join(self.get_discharge_frame().lazy(), on="hospitalization_id")
            .with_columns(
                tokens=pl.concat_list("adm_tokens", "tokens", "dis_tokens"),
                times=pl.concat_list("adm_times", "times", "dis_times"),
            )
            .select("hospitalization_id", "tokens", "times")
            .collect()
        )


if __name__ == "__main__":

    if os.uname().nodename.startswith("cri"):
        hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-development-sample")
    else:
        # change following line to develop locally
        hm = pathlib.Path("~/Documents/chicago/CLIF/clif-development-sample")

    out_dir = hm.parent.joinpath(hm.stem + "-tokenized").expanduser()
    out_dir.mkdir(exist_ok=True)

    tkzr = ClifTokenizer(data_dir=hm)
    tokens_timelines = tkzr.get_tokens_timelines()
    tokens_timelines.write_parquet(out_dir.joinpath("tokens_timelines.parquet"))
    tkzr.vocab.save(out_dir.joinpath("vocab.gzip"))

    tkzr2 = ClifTokenizer(data_dir=hm, vocab_path=out_dir.joinpath("vocab.gzip"))
    tokens_timelines2 = tkzr2.get_tokens_timelines()
    assert len(tkzr.vocab) == len(tkzr2.vocab)
    assert tkzr.vocab.lookup == tkzr2.vocab.lookup

    """tokenized summary stats
    """
    print("Timelines generated: {}".format(tokens_timelines.shape[0]))
    print("Vocabulary size: {}".format(len(tkzr.vocab)))
    print(
        "Summary stats of timeline lengths: \n {}".format(
            tokens_timelines.select(pl.col("tokens").list.len()).describe()
        )
    )
    for s in range(3):
        print(
            "Example timeline: \n {}".format(
                [
                    tkzr.vocab.reverse[t]
                    for t in tokens_timelines.sample(1, seed=s).select("tokens").item()
                ]
            )
        )
    print(
        "Summary stats of timeline duration: \n {}".format(
            tokens_timelines.select(
                pl.col("times").list.min().alias("start_time"),
                pl.col("times").list.max().alias("end_time"),
            )
            .select((pl.col("end_time") - pl.col("start_time")).alias("duration"))
            .describe()
        )
    )

    with pl.Config(tbl_rows=len(tkzr.vocab)):
        print(
            "Top 20 tokens by usage: \n {}".format(
                tokens_timelines.select("tokens")
                .explode("tokens")
                .rename({"tokens": "token"})
                .join(tkzr.vocab.get_frame(), on="token")
                .select("word")
                .to_series()
                .value_counts()
                .sort("count", descending=True)
                .head(20)
            )
        )
