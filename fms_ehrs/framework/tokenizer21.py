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

    def run_time_qc(self, reference_frame) -> Frame:
        return (
            reference_frame.join(
                pl.scan_parquet(
                    self.data_dir.joinpath(
                        f"{self.config['times_qc']['table']}.parquet"
                    )
                )
                .group_by(self.config["atomic_key"])
                .agg(
                    event_start_alt=pl.col(self.config["times_qc"]["dttm"]).min(),
                    event_end_alt=pl.col(self.config["times_qc"]["dttm"]).max(),
                ),
                how="left",
                on=self.config["atomic_key"],
                validate="1:1",
            )
            .with_columns(
                pl.min_horizontal(
                    self.config["reference"]["start_dttm"], "event_start_alt"
                )
                .cast(pl.Datetime(time_unit="ms"))
                .alias(self.config["reference"]["start_dttm"]),
                pl.max_horizontal(self.config["reference"]["end_dttm"], "event_end_alt")
                .cast(pl.Datetime(time_unit="ms"))
                .alias(self.config["reference"]["end_dttm"]),
            )
            .drop("event_start_alt", "event_end_alt")
            .filter(
                pl.col(self.config["reference"]["start_dttm"])
                < pl.col(self.config["reference"]["end_dttm"])
            )
        )

    def get_reference_frame(self) -> Frame:
        df = pl.scan_parquet(
            self.data_dir.joinpath(f"{self.config['reference']['table']}.parquet")
        )
        for tkv in self.config["augmentation_tables"]:
            df = df.join(
                pl.scan_parquet(self.data_dir.joinpath(f"{tkv['table']}.parquet")),
                on=tkv["key"],
                validate=tkv["validation"],
                how="left",
            )
        if "age" in self.config["reference"]:
            age = (
                df.select(self.config["reference"]["age"]).collect().to_numpy().ravel()
            )
            self.set_quants(
                v=age, c="AGE"
            )  # note this is a no-op if quants are already set for AGE
            df = df.with_columns(quantized_age=self.get_quants(v=age, c="AGE"))
        return self.run_time_qc(df)

    def get_end(self, end_type: typing.Literal["prefix", "suffix"]) -> Frame:
        time_col = (
            self.config["reference"]["start_dttm"]
            if end_type == "prefix"
            else self.config["reference"]["end_dttm"]
        )
        return self.get_reference_frame().select(
            pl.col(self.config["atomic_key"]).alias(self.config["atomic_key"]),
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
            pl.concat_list(
                ([self.vocab("TL_START")] if end_type == "prefix" else [])
                + [
                    (
                        pl.col(col["column"])
                        .str.to_lowercase()
                        .str.replace_all(" ", "_")
                        .map_elements(
                            lambda x, prefix=col["prefix"]: self.vocab(f"{prefix}_{x}"),
                            return_dtype=pl.Int64,
                            skip_nulls=False,
                        )
                        if not col["column"].startswith("quantized")
                        else pl.col(col["column"])
                    )
                    for col in self.config[end_type]
                ]
                + ([self.vocab("TL_END")] if end_type == "suffix" else [])
            ).alias("tokens"),
            pl.concat_list(
                [pl.col(time_col).cast(pl.Datetime(time_unit="ms"))]
                * (len(self.config[end_type]) + 1)
            ).alias("times"),
        )

    def get_event(
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
                    pl.col(self.config["atomic_key"]).alias(self.config["atomic_key"]),
                    pl.col(dttm).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
                    pl.col(category)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_")
                    .str.strip_chars(".")
                    .alias("category"),
                    pl.col(value).alias("value"),
                ).collect(),
                label=prefix,
            ).select(self.config["atomic_key"], "event_time", "tokens", "times")
        else:
            category_list = [category] if type(category) == str else category
            return df.select(
                pl.col(self.config["atomic_key"]).alias(self.config["atomic_key"]),
                pl.col(dttm).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
                pl.concat_list(
                    [
                        pl.col(cat)
                        .str.to_lowercase()
                        .str.replace_all(" ", "_")
                        .str.strip_chars(".")
                        .map_elements(
                            lambda x, prefix=prefix: self.vocab(f"{prefix}_{x}"),
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

    def get_raw_events(self) -> Frame:
        return pl.concat(self.get_event(**evt) for evt in self.config["events"])

    def get_tokens_timelines(self) -> Frame:

        # combine the admission tokens, event tokens, and discharge tokens
        tt = (
            self.get_end("prefix")
            .rename({"tokens": "prefix_tokens", "times": "prefix_times"})
            .join(
                self.get_events_frame(),
                on=self.config["atomic_key"],
                how="left",
                validate="1:1",
            )
            .join(
                self.get_end("suffix").rename(
                    {"tokens": "suffix_tokens", "times": "suffix_times"}
                ),
                on=self.config["atomic_key"],
                validate="1:1",
            )
            .with_columns(
                tokens=pl.concat_list("prefix_tokens", "tokens", "suffix_tokens"),
                times=pl.concat_list("prefix_times", "times", "suffix_times"),
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
            tt = self.cut_at_time(tt)

        return tt.collect()


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
