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
    tokenizes a directory containing a set of parquet files according to a
    configuration file
    """

    def __init__(
        self,
        *,
        data_dir: Pathlike = pathlib.Path("../.."),
        vocab_path: Pathlike = None,
        max_padded_len: int = None,
        quantizer: typing.Literal["centiles", "deciles", "sigmas", "ventiles"] = None,
        cut_at_24h: bool = False,
        include_time_spacing_tokens: bool = None,
        fused_category_values: bool = None,
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
                else self.config["options"].get("include_time_spacing_tokens", False)
            ),
            fused_category_values=(
                fused_category_values
                if fused_category_values is not None
                else self.config["options"].get("fused_category_values", False)
            ),
        )
        self.cut_at_24h: bool = cut_at_24h
        self.reference_frame = None

    def lazy_load(
        self,
        *,
        table: str = None,
        filter_expr: str | list = None,
        agg_expr: str | list = None,
        with_col_expr: str | list = None,
        key: str = None,
        subject_id_str: str = None,
        **kwargs,
    ) -> pl.LazyFrame:
        """lazy-load the table `table`.parquet and perform some standard ETL
        tasks in the following order if specified:
        1. fix subject_id to `subject_id_str`
        2. perform a filter operation
        3. perform an aggregation by `key` or self.config["subject_id"]
        4. add columns
        """
        df = pl.scan_parquet(self.data_dir / f"{table}.parquet")
        if subject_id_str is not None:
            df = df.with_columns(
                pl.col(subject_id_str).alias(self.config["subject_id"])
            )
        if filter_expr is not None:
            df = df.filter(
                eval(filter_expr)
                if isinstance(filter_expr, str)
                else [eval(c) for c in filter_expr]
            )
        if agg_expr is not None:
            df = df.group_by(key if key is not None else self.config["subject_id"]).agg(
                eval(agg_expr)
                if isinstance(agg_expr, str)
                else [eval(c) for c in agg_expr]
            )
        if with_col_expr is not None:
            df = df.with_columns(
                eval(with_col_expr)
                if isinstance(with_col_expr, str)
                else [eval(c) for c in with_col_expr]
            )
        return df

    def run_times_qc(self, reference_frame: Frame) -> Frame:
        return (
            (
                reference_frame.join(
                    self.lazy_load(table=self.config["times_qc"]["table"])
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
        """create the static reference frame as configured"""
        if self.reference_frame is not None:  # pull from cache if available
            return self.reference_frame
        df = self.lazy_load(**self.config["reference"])
        for tkv in self.config["augmentation_tables"]:
            df = df.join(
                self.lazy_load(**tkv),
                on=tkv["key"],
                validate=tkv["validation"],
                how="left",
                maintain_order="left",
            )
        if "age" in self.config["reference"]:
            age = (
                df.select(self.config["reference"]["age"]).collect().to_numpy().ravel()
            )
            self.set_quants(
                v=age, c="AGE"
            )  # note this is a no-op if quants are already set for AGE
            df = df.with_columns(quantized_age=self.get_quants(v=age, c="AGE"))
            if self.fused_category_values:
                df = df.with_columns(
                    pl.col("quantized_age").map_elements(
                        lambda x: self.vocab(f"AGE_Q{x}"),
                        return_dtype=pl.Int64,
                        skip_nulls=False,
                    )
                )
        self.reference_frame = self.run_times_qc(df)  # save to cache
        return self.reference_frame

    def get_end(self, end_type: typing.Literal["prefix", "suffix"]) -> Frame:
        """create the prefix or suffix tokens"""
        time_col = (
            self.config["reference"]["start_time"]
            if end_type == "prefix"
            else self.config["reference"]["end_time"]
        )
        df = self.get_reference_frame().with_columns(
            pl.concat_list(
                ([self.vocab("TL_START")] if end_type == "prefix" else [])
                + [
                    (
                        (
                            pl.col(col["column"])
                            .str.replace_all(" ", "_")
                            .map_elements(
                                lambda x, prefix=col["prefix"]: self.vocab(
                                    f"{prefix}_{x}"
                                ),
                                return_dtype=pl.Int64,
                                skip_nulls=False,
                            )
                            if not col["column"].startswith("quantized")
                            else pl.col(col["column"])
                        )
                        if "is_list" not in col or not col["is_list"]
                        else pl.col(col["column"]).map_elements(
                            lambda x, prefix=col["prefix"]: [
                                self.vocab(f"{prefix}_{y}") for y in x
                            ],
                            return_dtype=pl.List(pl.Int64),
                        )
                    )
                    for col in self.config[end_type]
                ]
                + ([self.vocab("TL_END")] if end_type == "suffix" else [])
            ).alias("tokens")
        )
        return df.select(
            self.config["subject_id"],
            pl.col(time_col).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
            "tokens",
            pl.col(time_col)
            .cast(pl.Datetime(time_unit="ms"))
            .repeat_by(pl.col("tokens").list.len())
            .alias("times"),
        )

    def get_event(
        self,
        table: str,
        time: str,
        code: str,
        *,
        prefix: str = None,
        numeric_value: str = None,
        text_value: str = None,
        filter_expr: str = None,
        with_col_expr: str = None,
        reference_key: str = None,
        subject_id_str: str = None,
        fix_date_to_time: bool = None,
    ) -> Frame | None:
        """create tokens corresponding to a configured event"""
        df = self.lazy_load(
            table=table,
            filter_expr=filter_expr,
            with_col_expr=with_col_expr,
            subject_id_str=subject_id_str,
        )
        if (
            fix_date_to_time is not None and bool(fix_date_to_time)
        ):  # if a date was cast to a time, the default of 00:00:00 should be replaced with 23:59:59
            df = df.with_columns(
                pl.col(time)
                .cast(pl.Datetime(time_unit="ms"))
                .dt.replace(hour=23, minute=59, second=59)
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
            df_cv = df.select(
                pl.col(self.config["subject_id"]),
                pl.col(time).cast(pl.Datetime(time_unit="ms")).alias("event_time"),
                pl.col(code)
                .cast(str)
                .str.to_lowercase()
                .str.replace_all(" ", "_")
                .str.strip_chars(".")
                .alias("category"),
                pl.col(numeric_value).alias("value"),
            ).collect()
            return (
                self.process_cat_val_frame(df_cv, label=prefix).select(
                    self.config["subject_id"], "event_time", "tokens"
                )
                if len(df_cv) > 0
                else None
            )
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
            ).collect()

    def get_events(self) -> Frame:
        """create all events as configured"""
        event_tokens = (
            pl.concat(
                e
                for evt in self.config["events"]
                if (e := self.get_event(**evt)) is not None
            )
            .lazy()
            .filter(pl.col("event_time").is_not_null())
            .sort("event_time", pl.col("tokens").list.first())
            .explode("tokens")
            .filter(pl.col("tokens") != self.vocab(None))
            .group_by(self.config["subject_id"], maintain_order=True)
            .agg(tokens=pl.col("tokens"), times=pl.col("event_time").alias("times"))
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
        """combine the prefix tokens, event tokens, and suffix tokens"""
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
                tokens=pl.concat_list(
                    "prefix_tokens", pl.col("tokens").fill_null([]), "suffix_tokens"
                ),
                times=pl.concat_list(
                    "prefix_times", pl.col("times").fill_null([]), "suffix_times"
                ),
            )
            .select(self.config["subject_id"], "tokens", "times")
            .sort(by=self.config["subject_id"])
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
    # import tempfile

    dev_dir = (
        pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/development-sample-21")
        if os.uname().nodename.startswith("cri")
        else pathlib.Path("~/Downloads/development-sample-21")
        .expanduser()
        .resolve()  # change if developing locally
    )

    df = pl.read_parquet(dev_dir / "raw-meds-ed" / "dev" / "meds.parquet")

    tkzr_meds_ed = Tokenizer21(
        config_file="../config/config-mimic-meds-ed.yaml",
        data_dir=dev_dir / "raw-meds-ed" / "dev",
    )

    x = tkzr_meds_ed.get_reference_frame().collect()
    print(x)

    tt_meds_ed = tkzr_meds_ed.get_tokens_timelines()
    summarize(tkzr_meds_ed, tt_meds_ed)
    tkzr_meds_ed.vocab.print_aux()
    print(list(tkzr_meds_ed.vocab.lookup.keys()))

    # tkzr21_pp = Tokenizer21(
    #     config_file="../config/config-21++.yaml",
    #     data_dir=dev_dir.joinpath("raw-mimic/dev"),
    # )
    # tt21_pp = tkzr21_pp.get_tokens_timelines()
    # summarize(tkzr21_pp, tt21_pp)
    # print(f"{len(tkzr21_pp.vocab)=}")
    # tkzr21_pp.vocab.print_aux()
    # print(list(tkzr21_pp.vocab.lookup.keys()))

    # with tempfile.NamedTemporaryFile() as fp:
    #     tkzr21_pp.vocab.save(fp.name)
    #     tkzr21_pp_ucmc = Tokenizer21(
    #         vocab_path=fp.name,
    #         config_file="../config/config-21++.yaml",
    #         data_dir=dev_dir.joinpath("raw-ucmc/dev"),
    #     )
    #     tt21_pp_ucmc = tkzr21_pp_ucmc.get_tokens_timelines()
    #     summarize(tkzr21_pp_ucmc, tt21_pp_ucmc)
