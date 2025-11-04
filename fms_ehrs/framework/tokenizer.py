#!/usr/bin/env python3

"""
provides a configurable tokenizer object to take tabular data and convert
it to tokenized timelines at the subject_id level
"""

import os
import pathlib
import typing

import polars as pl
import ruamel.yaml as yaml

from fms_ehrs.framework.tokenizer_base import BaseTokenizer, summarize

Frame: typing.TypeAlias = pl.DataFrame | pl.LazyFrame
Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike


class Tokenizer21(BaseTokenizer):
    """
    tokenizes a directory containing a set of parquet files corresponding to
    the CLIF-2.1 standard
    """

    def __init__(
        self,
        *,
        data_dir: Pathlike = pathlib.Path("../.."),
        vocab_path: Pathlike = None,
        max_padded_len: int = None,
        quantizer: typing.Literal["deciles", "sigmas"] = None,
        cut_at_24h: bool = False,
        include_time_spacing_tokens: bool = None,
        config_file: Pathlike = None,
    ):
        self.config = yaml.YAML(typ="safe").load(
            pathlib.Path(config_file).expanduser().resolve()
        )
        super().__init__(
            data_dir=data_dir,
            vocab_path=vocab_path,
            max_padded_len=(
                max_padded_len
                if max_padded_len is not None
                else self.config["options"]["max_padded_len"]
            ),  # passed argument overrides config default
            quantizer=(
                quantizer
                if quantizer is not None
                else self.config["options"]["quantizer"]
            ),
            include_time_spacing_tokens=(
                include_time_spacing_tokens
                if include_time_spacing_tokens is not None
                else self.config["options"]["include_time_spacing_tokens"]
            ),
        )
        self.cut_at_24h: bool = cut_at_24h
        self.reference_frame = None

    def run_times_qc(self, reference_frame) -> Frame:
        return (
            (
                reference_frame.join(
                    pl.scan_parquet(
                        self.data_dir.joinpath(
                            f"{self.config['times_qc']['table']}.parquet"
                        )
                    )
                    .group_by(self.config["subject_id"])
                    .agg(
                        start_time_alt=pl.col(self.config["times_qc"]["time"]).min(),
                        end_time_alt=pl.col(self.config["times_qc"]["time"]).max(),
                    ),
                    how="left",
                    on=self.config["subject_id"],
                    validate="1:1",
                )
                .with_columns(
                    pl.min_horizontal(
                        self.config["reference"]["start_time"], "start_time_alt"
                    )
                    .cast(pl.Datetime(time_unit="ms"))
                    .alias(self.config["reference"]["start_time"]),
                    pl.max_horizontal(
                        self.config["reference"]["end_time"], "end_time_alt"
                    )
                    .cast(pl.Datetime(time_unit="ms"))
                    .alias(self.config["reference"]["end_time"]),
                )
                .drop("start_time_alt", "end_time_alt")
                .filter(
                    pl.col(self.config["reference"]["start_time"])
                    < pl.col(self.config["reference"]["end_time"])
                )
            )
            if "times_qc" in self.config
            else reference_frame
        )

    def get_reference_frame(self) -> Frame:
        if self.reference_frame is not None:
            return self.reference_frame
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
        self.reference_frame = self.run_times_qc(df)
        return self.reference_frame

    def get_end(self, end_type: typing.Literal["prefix", "suffix"]) -> Frame:
        time_col = (
            self.config["reference"]["start_time"]
            if end_type == "prefix"
            else self.config["reference"]["end_time"]
        )
        return self.get_reference_frame().select(
            pl.col(self.config["subject_id"]),
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
        *,
        prefix: str = None,
        time: str,
        code: str,
        numeric_value: str = None,
        text_value: str = None,
        filter: str = None,
        reference_key: str = None,
    ):
        df = pl.scan_parquet(self.data_dir.joinpath(f"{table}.parquet")).filter(
            eval(filter) if filter is not None else True
        )
        if reference_key is not None:
            df = df.join(self.reference_frame, on=reference_key, how="inner").filter(
                pl.col(time)
                .cast(pl.Datetime(time_unit="ms"))
                .is_between(
                    self.config["reference"]["start_time"],
                    self.config["reference"]["end_time"],
                )
            )
        if numeric_value is not None:
            # pass to category-value tokenizer
            return self.process_cat_val_frame(
                df.select(
                    pl.col(self.config["subject_id"]),
                    pl.col(time).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
                    pl.col(code)
                    .cast(str)
                    .str.to_lowercase()
                    .str.replace_all(" ", "_")
                    .str.strip_chars(".")
                    .alias("category"),
                    pl.col(numeric_value).alias("value"),
                ).collect(),
                label=prefix,
            ).select(self.config["subject_id"], "event_time", "tokens", "times")
        else:
            category_list = [code] if isinstance(code, str) else code
            if text_value is not None:
                category_list += (
                    [text_value] if isinstance(text_value, str) else text_value
                )
            # tokenize provided categories directly
            return df.select(
                pl.col(self.config["subject_id"]),
                pl.col(time).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
                pl.concat_list(
                    [
                        pl.col(cat)
                        .cast(str)
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
                    [pl.col(time).cast(pl.Datetime("ms"))] * len(category_list)
                ).alias("times"),
            ).collect()

    def get_events(self) -> Frame:
        event_tokens = (
            pl.concat(self.get_event(**evt) for evt in self.config["events"])
            .lazy()
            .sort("event_time", pl.col("tokens").list.first())
            .group_by("hospitalization_id", maintain_order=True)
            .agg(tokens=pl.col("tokens").explode(), times=pl.col("times").explode())
        )

        if self.include_time_spacing_tokens:
            event_tokens = (
                event_tokens.with_columns(
                    pl.struct(["tokens", "times"])
                    .map_elements(
                        lambda x: self.time_spacing_inserter(x["tokens"], x["times"]),
                        return_dtype=pl.Struct(
                            [
                                pl.Field("tokens", pl.List(pl.Int64)),
                                pl.Field("times", pl.List(pl.Datetime())),
                            ]
                        ),
                    )
                    .alias("inserted")
                )
                .with_columns(
                    pl.col("inserted").struct.field("tokens").alias("tokens"),
                    pl.col("inserted")
                    .struct.field("times")
                    .cast(pl.List(pl.Datetime(time_unit="ms")))
                    .alias("times"),
                )
                .drop("inserted")
            )
        return event_tokens

    def get_tokens_timelines(self) -> Frame:
        # combine the prefix tokens, event tokens, and suffix tokens
        tt = (
            self.get_end("prefix")
            .rename({"tokens": "prefix_tokens", "times": "prefix_times"})
            .join(
                self.get_events(),
                on=self.config["subject_id"],
                how="left",
                validate="1:1",
            )
            .join(
                self.get_end("suffix").rename(
                    {"tokens": "suffix_tokens", "times": "suffix_times"}
                ),
                on=self.config["subject_id"],
                how="left",
                validate="1:1",
            )
            .with_columns(
                tokens=pl.concat_list("prefix_tokens", "tokens", "suffix_tokens"),
                times=pl.concat_list("prefix_times", "times", "suffix_times"),
            )
            .select("hospitalization_id", "tokens", "times")
            .sort(by="hospitalization_id")
        )

        if self.config["options"]["day_stay_filter"]:
            tt = tt.filter(
                (pl.col("times").list.get(-1) - pl.col("times").list.get(0))
                >= pl.duration(days=1)
            )

        if self.cut_at_24h:
            tt = super().cut_at_time(tt)

        return tt.collect()


if __name__ == "__main__":
    dev_dir = (
        pathlib.Path("~/Documents/chicago/CLIF/development-sample/raw-21")
        .expanduser()
        .resolve()
    )

    # tkzr20 = Tokenizer21(config_file="../config/config-20.yaml", data_dir=dev_dir)
    # tt20 = tkzr20.get_tokens_timelines()
    # summarize(tkzr20, tt20)

    tkzr21 = Tokenizer21(
        config_file="../config/config-21.yaml",
        data_dir=dev_dir,
        include_time_spacing_tokens=True,
    )
    tt21 = tkzr21.get_tokens_timelines()
    summarize(tkzr21, tt21)

    x = pl.read_parquet(dev_dir.joinpath("clif_medication_admin_intermittent.parquet"))
    with pl.Config(tbl_cols=-1):
        print(x)

    pl.read_parquet(
        dev_dir.joinpath("clif_medication_admin_intermittent.parquet")
    ).select("mar_action_name").to_series().value_counts().sort(
        "count", descending=True
    )

    print(
        tt21.with_columns(
            prefix=pl.col("tokens")
            .list.head(n=6)
            .list.eval(
                pl.element().map_elements(
                    lambda v: tkzr21.vocab.reverse[v], return_dtype=pl.String
                )
            )
            .list.join(", "),
            suffix=pl.col("tokens")
            .list.tail(2)
            .list.eval(
                pl.element().map_elements(
                    lambda v: tkzr21.vocab.reverse[v], return_dtype=pl.String
                )
            )
            .list.join(", "),
        )
    )
