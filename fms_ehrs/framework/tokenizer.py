#!/usr/bin/env python3

"""
provides a simple tokenizing interface to take tabular CLIF data and convert
it to tokenized timelines at the hospitalization_id level
"""

import collections
import functools
import os
import pathlib
import re
import typing

import polars as pl

from fms_ehrs.framework.tokenizer0 import BaseTokenizer, summarize

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class ClifTokenizer(BaseTokenizer):
    """
    tokenizes a directory containing a set of parquet files corresponding to
    the CLIF-2.0 standard; note that the `cut_at_24h` flag implements a very
    conservative cut and typically removes some timelines
    """

    def __init__(
        self,
        *,
        data_dir: Pathlike = pathlib.Path("../.."),
        vocab_path: Pathlike = None,
        max_padded_len: int = None,
        day_stay_filter: bool = False,
        cut_at_24h: bool = False,
        valid_admission_window: tuple[str, str] = None,
        lab_time: typing.Literal["collect", "result"] = "result",
        quantizer: typing.Literal["deciles", "sigmas"] = "deciles",
        drop_deciles: bool = False,
        drop_nulls_nans: bool = False,
        n_top_reports: int = 100,
        include_time_spacing_tokens: bool = False,
    ):
        """
        if no vocabulary is provided, we are in training mode; otherwise, the
        provided vocabulary is frozen
        """
        super().__init__(
            data_dir=data_dir,
            vocab_path=vocab_path,
            max_padded_len=max_padded_len,
            quantizer=quantizer,
            include_time_spacing_tokens=include_time_spacing_tokens,
        )
        self.tbl = dict()
        self.day_stay_filter = bool(day_stay_filter)
        self.cut_at_24h = bool(cut_at_24h)
        self.valid_admission_window = valid_admission_window
        self.lab_time = lab_time
        self.drop_deciles = bool(drop_deciles)
        self.drop_nulls_nans = bool(drop_nulls_nans)
        self.n_top_reports = n_top_reports
        self.include_time_spacing_tokens = bool(include_time_spacing_tokens)

    def load_tables(self) -> None:
        """lazy-load all parquet tables from the directory `self.data_dir`"""
        self.tbl = {
            (
                p.stem.split("_")[1] if "assessments" not in p.stem else "assessments"
            ): pl.scan_parquet(p)
            for p in self.data_dir.glob("*.parquet")
        }

    def process_tables(self) -> None:
        self.tbl["patient"] = (
            self.tbl["patient"]
            .group_by("patient_id")
            .agg(
                pl.col("race_category")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .first(),
                pl.col("ethnicity_category")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .first(),
                pl.col("sex_category")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .first(),
            )
            .with_columns(
                pl.col("race_category").map_elements(
                    lambda x: self.vocab("RACE_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
                pl.col("ethnicity_category").map_elements(
                    lambda x: self.vocab("ETHN_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
                pl.col("sex_category").map_elements(
                    lambda x: self.vocab("SEX_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
            )
            .with_columns(
                tokens=pl.concat_list(
                    "race_category", "ethnicity_category", "sex_category"
                )
            )
            .select("patient_id", "tokens")
            .collect()
        )

        self.tbl["hospitalization"] = (
            self.tbl["hospitalization"]
            .group_by("hospitalization_id")
            .agg(
                pl.col("patient_id").first(),
                pl.col("admission_dttm")
                .first()
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_start"),
                pl.col("discharge_dttm")
                .first()
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_end"),
                pl.col("age_at_admission").first(),
                pl.col("admission_type_name")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .first(),
                pl.col("discharge_category")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .first(),
            )
            .filter(
                pl.col("event_start").is_between(
                    pl.lit(self.valid_admission_window[0]).cast(pl.Date),
                    pl.lit(self.valid_admission_window[1]).cast(pl.Date),
                )
                if self.valid_admission_window is not None
                else True
            )
            .with_columns(
                pl.col("admission_type_name").map_elements(
                    lambda x: self.vocab("ADMN_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
                pl.col("discharge_category").map_elements(
                    lambda x: self.vocab("DSCG_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
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
            .sort(by="hospitalization_id")
            .collect()
        )

        # tokenize age_at_admission here
        c = "age_at_admission"
        v = self.tbl["hospitalization"].select(c).to_numpy().ravel()
        self.set_quants(v=v, c=c)
        self.tbl["hospitalization"] = (
            self.tbl["hospitalization"]
            .with_columns(age_at_admission=self.get_quants(v=v, c=c))
            .with_columns(admission_tokens=pl.concat_list(c, "admission_type_name"))
            .drop(c, "admission_type_name")
        )

        self.tbl["adt"] = (
            self.tbl["adt"]
            .with_columns(
                event_time=pl.col("in_dttm").cast(pl.Datetime(time_unit="ms")),
                category=pl.col("location_category").str.to_lowercase(),
            )
            .with_columns(
                tokens=pl.col("category")
                .str.to_lowercase()
                .map_elements(
                    lambda x: [self.vocab("ADT_{}".format(x))],
                    return_dtype=pl.List(pl.Int64),
                    skip_nulls=False,
                ),
                times=pl.concat_list([pl.col("event_time").cast(pl.Datetime("ms"))]),
            )
            .select("hospitalization_id", "event_time", "tokens", "times")
            .collect()
        )

        self.tbl["labs"] = (
            self.tbl["labs"]
            .filter(~pl.col("lab_category").is_null())
            .select(
                "hospitalization_id",
                pl.col(f"lab_{self.lab_time}_dttm")
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_time"),
                pl.col("lab_category").str.to_lowercase().alias("category"),
                pl.col("lab_value_numeric").alias("value"),
            )
            .collect()
        )
        self.tbl["labs"] = self.process_cat_val_frame(self.tbl["labs"], label="LAB")

        self.tbl["vitals"] = (
            self.tbl["vitals"]
            .select(
                "hospitalization_id",
                pl.col("recorded_dttm")
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_time"),
                pl.col("vital_category")
                .cast(pl.String)
                .str.to_lowercase()
                .alias("category"),
                pl.col("vital_value").alias("value"),
            )
            .collect()
        )
        self.tbl["vitals"] = self.process_cat_val_frame(self.tbl["vitals"], label="VTL")

        self.tbl["medication"] = (
            self.tbl["medication"]
            .select(
                "hospitalization_id",
                pl.col("admin_dttm")
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_time"),
                pl.col("med_category").str.to_lowercase().alias("category"),
                pl.col("med_dose").alias("value"),
            )
            .collect()
        )
        self.tbl["medication"] = self.process_cat_val_frame(
            self.tbl["medication"], label="MED"
        )

        # seems like there's a column for assessment, and then either a
        # numerical_value OR a categorical_value, depending on the assessment
        self.tbl["assessments"] = (
            self.tbl["assessments"]
            .with_columns(
                event_time=pl.col("recorded_dttm").cast(pl.Datetime(time_unit="ms")),
                category=pl.col("assessment_category").str.to_lowercase(),
                value=pl.col("numerical_value"),
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
                pl.col("category")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .map_elements(
                    lambda x: self.vocab("ASMT_cat_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
                pl.col("categorical_value")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .map_elements(
                    lambda x: self.vocab("ASMT_val_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
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
            .with_columns(
                pl.col("mode_category")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .map_elements(
                    lambda x: self.vocab("RESP_mode_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
                pl.col("device_category")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .map_elements(
                    lambda x: self.vocab("RESP_devc_{}".format(x)),
                    return_dtype=pl.Int64,
                    skip_nulls=False,
                ),
                event_time=pl.col("recorded_dttm").cast(pl.Datetime(time_unit="ms")),
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
            .filter(pl.col("position_category") == "prone")
            .with_columns(
                event_time=pl.col("recorded_dttm").cast(pl.Datetime(time_unit="ms"))
            )
            .with_columns(
                tokens=pl.col("position_category").map_elements(
                    lambda x: [self.vocab("POSN_{}".format(x))],
                    return_dtype=pl.List(pl.Int64),
                    skip_nulls=False,
                ),
                times=pl.concat_list([pl.col("event_time").cast(pl.Datetime("ms"))]),
            )
            .collect()
        )

        # process machine measurements from ECG's if available
        if "measurements" in self.tbl:
            self.tbl["measurements"] = self.tbl["measurements"].with_columns(
                reports=pl.concat_list(
                    *[
                        pl.col(f"report_{i}").str.strip_chars(" .").str.to_uppercase()
                        for i in range(18)
                    ]
                ).list.eval(pl.element().drop_nulls()),
                event_time=pl.col("event_dttm").cast(pl.Datetime(time_unit="ms")),
            )

            if (
                not self.vocab.has_aux("ECG_machine_measurements")
                and self.vocab.is_training
            ):
                self.vocab.set_aux(
                    "ECG_machine_measurements",
                    set(
                        self.tbl["measurements"]
                        .select(pl.col("reports").explode())
                        .group_by("reports")
                        .len()
                        .sort("len")
                        .tail(self.n_top_reports)
                        .collect()
                        .to_series()
                        .to_list()
                    ),
                )

            self.tbl["measurements"] = (
                self.tbl["measurements"]
                .with_columns(
                    pl.col("reports")
                    .list.eval(
                        pl.element().filter(
                            pl.element().is_in(
                                self.vocab.get_aux("machine_measurements")
                            )
                        )
                    )
                    .list.eval(
                        pl.element().map_elements(
                            lambda x: self.vocab("ECG_{}".replace(" ", "_").format(x)),
                            return_dtype=pl.Int64,
                        )
                    )
                    .alias("tokens")
                )
                .with_columns(
                    pl.struct(["event_time", pl.col("tokens").list.len()])
                    .map_elements(
                        lambda row: [row["event_time"]] * row["tokens"],
                        return_dtype=pl.List(pl.Datetime),
                    )
                    .alias("times")
                )
                .cast({"times": pl.List(pl.Datetime(time_unit="ms"))})
                .collect()
            )

    def run_times_qc(self) -> None:
        alt_times = (
            self.tbl["vitals"]
            .group_by("hospitalization_id")
            .agg(
                event_start_alt=pl.col("event_time").min(),
                event_end_alt=pl.col("event_time").max(),
            )
        )

        self.tbl["hospitalization"] = (
            self.tbl["hospitalization"]
            .join(alt_times, how="left", on="hospitalization_id", validate="1:1")
            .with_columns(
                event_start=pl.min_horizontal("event_start", "event_start_alt"),
                event_end=pl.max_horizontal("event_end", "event_end_alt"),
            )
            .drop("event_start_alt", "event_end_alt")
            .filter(pl.col("event_start") < pl.col("event_end"))
        )

    def get_admission_frame(self) -> Frame:
        ## prepend patient-level tokens to each admission event
        admission_tokens = (
            self.tbl["patient"]
            .join(self.tbl["hospitalization"], on="patient_id", validate="1:m")
            .cast({"event_start": pl.Datetime(time_unit="ms")})
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
            .cast({"event_time": pl.Datetime(time_unit="ms")})
            .with_columns(
                dis_tokens=pl.concat_list(
                    "discharge_category", pl.lit(self.vocab("TL_END"))
                ),
                dis_times=pl.concat_list(*[pl.col("event_time")] * 2),
            )
            .select("hospitalization_id", "event_time", "dis_tokens", "dis_times")
        )

        return discharge_tokens

    def get_raw_events(self) -> Frame:
        return pl.concat(
            self.tbl[k].select("hospitalization_id", "event_time", "tokens", "times")
            for k in self.tbl.keys()
            if k not in ("patient", "hospitalization")
        )

    def get_events_frame(self) -> Frame:
        events = self.get_raw_events()

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
            .agg([pl.col("times").explode()])
        )

        event_tokens = tokens_agg.join(
            times_agg, on="hospitalization_id", validate="1:1", maintain_order="left"
        )

        if self.include_time_spacing_tokens:
            event_tokens = event_tokens.with_columns(
                pl.struct(["tokens", "times"])
                .map_elements(
                    lambda x: self.time_spacing_inserter(x["tokens"], x["times"])[
                        "tokens"
                    ],
                    return_dtype=pl.List(pl.Int64),
                )
                .alias("tokens"),
                pl.struct(["tokens", "times"])
                .map_elements(
                    lambda x: self.time_spacing_inserter(x["tokens"], x["times"])[
                        "times"
                    ],
                    return_dtype=pl.List(pl.Datetime(time_unit="ms")),
                )
                .alias("times"),
            )
        return event_tokens

    def get_tokens_timelines(self) -> Frame:
        self.load_tables()
        self.process_tables()
        self.run_times_qc()

        # combine the admission tokens, event tokens, and discharge tokens
        tt = (
            self.get_admission_frame()
            .lazy()
            .join(
                self.get_events_frame(),
                on="hospitalization_id",
                how="left",
                validate="1:1",
            )
            .join(
                self.get_discharge_frame().lazy(),
                on="hospitalization_id",
                validate="1:1",
            )
            .with_columns(
                tokens=pl.concat_list("adm_tokens", "tokens", "dis_tokens"),
                times=pl.concat_list("adm_times", "times", "dis_times"),
            )
            .select("hospitalization_id", "tokens", "times")
            .sort(by="hospitalization_id")
        )

        if self.day_stay_filter:
            tt = tt.filter(
                (pl.col("times").list.get(-1) - pl.col("times").list.get(0))
                >= pl.duration(days=1)
            )

        if self.cut_at_24h:
            tt = super().cut_at_time(tt)

        if self.drop_deciles or self.drop_nulls_nans:
            filtered = (
                tt.explode("tokens", "times")
                .filter(pl.col("tokens") >= 10 if self.drop_deciles else pl.lit(True))
                .filter(
                    (~pl.col("tokens").is_in([self.vocab(None), self.vocab("nan")]))
                    if self.drop_nulls_nans
                    else pl.lit(True)
                )
            )
            new_times = filtered.group_by(
                "hospitalization_id", maintain_order=True
            ).agg([pl.col("times")])
            new_tokens = filtered.group_by(
                "hospitalization_id", maintain_order=True
            ).agg([pl.col("tokens")])
            tt = new_tokens.join(
                new_times,
                on="hospitalization_id",
                validate="1:1",
                maintain_order="left",
            )

        return tt.collect()

    def pad_and_truncate(self, tokens_timelines: Frame) -> Frame:
        if self.max_padded_length is not None:
            tt = tokens_timelines.lazy().with_columns(
                seq_len=pl.col("tokens").list.len()
            )
            tt_under = tt.filter(
                pl.col("seq_len") <= self.max_padded_length
            ).with_columns(
                padded=pl.concat_list(
                    "tokens",
                    pl.lit(self.vocab("PAD")).repeat_by(
                        self.max_padded_length - pl.col("seq_len")
                    ),
                )
            )
            tt_over = tt.filter(
                pl.col("seq_len") > self.max_padded_length
            ).with_columns(
                padded=pl.concat_list(
                    pl.col("tokens").list.slice(
                        offset=0, length=self.max_padded_length - 1
                    ),
                    pl.lit(self.vocab("TRUNC")),
                )
            )
            return pl.concat([tt_under, tt_over]).collect()
        else:
            return tokens_timelines

    def print_aux(self) -> None:
        self.vocab.print_aux()


@functools.cache
def token_type(word: str) -> str:
    if word in ClifTokenizer().special:
        return "SPECIAL"
    elif re.fullmatch(r"Q\d", word) or re.fullmatch(r"Q[0-3][+-]", word):
        return "Q"
    else:
        return word.split("_")[0]


type_names = collections.OrderedDict(
    Q="Q",
    RACE="RACE",
    ETHN="ETHNICITY",
    SEX="SEX",
    ADMN="ADMISSION",
    ADT="TRANSFER",
    ASMT="ASSESSMENT",
    LAB="LAB",
    MED="MEDICATION",
    POSN="POSITION",
    RESP="RESPIRATION",
    VTL="VITALS",
    DSCG="DISCHARGE",
    SPECIAL="SPECIAL",
)
token_types = tuple(type_names.keys())

if __name__ == "__main__":
    if os.uname().nodename.startswith("cri"):
        hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-development-sample")
    else:
        # change following line to develop locally
        hm = pathlib.Path("~/Documents/chicago/CLIF/development-sample/raw")

    out_dir = hm.parent.joinpath(hm.stem + "-tokenized").expanduser()
    out_dir.mkdir(exist_ok=True)

    tkzr = ClifTokenizer(
        data_dir=hm,
        max_padded_len=1024,
        day_stay_filter=True,
        cut_at_24h=True,
        valid_admission_window=("2110-01-01", "2111-12-31"),
        drop_nulls_nans=True,
        # include_time_spacing_tokens=True,
    )
    tt = tokens_timelines = tkzr.get_tokens_timelines()

    tkzr.print_aux()
    summarize(tkzr, tokens_timelines)

    tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
    tokens_timelines.write_parquet(out_dir.joinpath("tokens_timelines.parquet"))
    tkzr.vocab.save(out_dir.joinpath("vocab.gzip"))

    tkzr2 = ClifTokenizer(data_dir=hm, vocab_path=out_dir.joinpath("vocab.gzip"))
    tokens_timelines2 = tkzr2.get_tokens_timelines()
    assert len(tkzr.vocab) == len(tkzr2.vocab)
    assert tkzr.vocab.lookup == tkzr2.vocab.lookup