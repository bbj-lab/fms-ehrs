#!/usr/bin/env python3

"""
provides a simple tokenizing interface to take tabular data and convert
it to tokenized timelines at the hospitalization_id level
"""

import os
import pathlib
import typing

import polars as pl
import ruamel.yaml as yaml

from fms_ehrs.framework.tokenizer import ClifTokenizer, summarize

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class Tokenizer21(ClifTokenizer):
    """
    tokenizes a directory containing a set of parquet files corresponding to
    the CLIF-2.1 standard
    """

    def __init__(self, config_file: Pathlike = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = yaml.YAML(typ="safe").load(
            pathlib.Path(config_file).expanduser().resolve()
        )

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

        self.tbl["vitals"] = (
            self.tbl["vitals"]
            .select(
                "hospitalization_id",
                pl.col("recorded_dttm")
                .cast(pl.Datetime(time_unit="ms"))
                .alias("event_time"),
            )
            .collect()
        )

    def gather_event(
        self,
        table: str,
        prefix: str,
        category: str,
        value: str,
        dttm: str,
        filter: str = None,
    ):
        df = pl.scan_parquet(self.data_dir.joinpath(f"{table}.parquet")).filter(
            eval(filter) if filter is not None else True
        )
        if value is not None:
            return self.process_cat_val_frame(
                df.select(
                    pl.col(self.config["atomic_key"]).alias("hospitalization_id"),
                    pl.col(dttm).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
                    pl.col(category)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_")
                    .alias("category"),
                    pl.col(value).alias("value"),
                ).collect(),
                label=prefix,
            ).select("hospitalization_id", "event_time", "tokens", "times")
        else:
            category_list = [category] if type(category) == str else category
            return df.select(
                pl.col(self.config["atomic_key"]).alias("hospitalization_id"),
                pl.col(dttm).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
                pl.concat_list(
                    [
                        pl.col(cat)
                        .str.to_lowercase()
                        .str.replace_all(" ", "_")
                        .map_elements(
                            lambda x: self.vocab(f"{prefix}_{x}"),
                            return_dtype=pl.Int64,
                            skip_nulls=False,
                        )
                        for cat in category_list
                    ]
                ).alias("tokens"),
                pl.concat_list(
                    [pl.col(dttm).cast(pl.Datetime("ms"))] * len(category_list)
                ).alias("times"),
            ).collect()

    def collect_raw_events(self) -> Frame:
        return pl.concat(self.gather_event(**evt) for evt in self.config["events"])


if __name__ == "__main__":
    tkzr_old = ClifTokenizer(data_dir="~/Documents/chicago/CLIF/development-sample/raw")
    tt_old = tkzr_old.get_tokens_timelines()
    summarize(tkzr_old, tt_old)

    tkzr_new = Tokenizer21(
        config_file="../../config-20.yaml",
        data_dir="~/Documents/chicago/CLIF/development-sample/raw",
    )
    tt_new = tkzr_new.get_tokens_timelines()
    summarize(tkzr_new, tt_new)
