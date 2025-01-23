#!/usr/bin/env python3

import gzip
import os
import pathlib
import pickle

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

type Frame = pl.DataFrame | pl.LazyFrame


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
        self._is_training = True

    def __call__(self, word: str) -> int | None:
        try:
            return self.lookup[word]
        except KeyError:
            if self._is_training:
                self.lookup[word], self.reverse[n] = (n := len(self.lookup)), word
                return n
            else:
                return None

    def set_aux(self, word: str, aux_data):
        self.aux[word] = aux_data

    def has_aux(self, word: str):
        return word in self.aux

    def in_lookup(self, word: str):
        return word in self.lookup

    def get_aux(self, word: str):
        return self.aux[word]

    def save(self, filepath: pathlib.PurePath | str):
        with gzip.open(pathlib.Path(filepath).expanduser(), "w+") as f:
            pickle.dump(
                {
                    "lookup": self.lookup,
                    "reverse": self.reverse,
                    "aux": {k: list(v) for k, v in self.aux.items()},
                },
                f,
            )

    def load(self, filepath: pathlib.PurePath | str):
        with gzip.open(pathlib.Path(filepath).expanduser(), mode="r+") as f:
            for k, v in pickle.load(f).items():
                setattr(self, k, v)

    def get_frame(self) -> Frame:
        return pl.from_records(
            list(self.lookup.items()), schema=("word", "token"), orient="row"
        )

    def __len__(self) -> int:
        return len(self.lookup)

    @property
    def is_training(self):
        return self._is_training

    @is_training.setter
    def is_training(self, value: bool):
        self._is_training = value


def process_single_category(x: Frame, label: str, vocab: Vocabulary) -> Frame:
    """Quantize a sub-table consisting of a single category

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
    if not vocab.has_aux(f"{label}_{c}"):
        vocab.set_aux(f"{label}_{c}", np.nanquantile(v, np.arange(0.1, 1.0, 0.1)))
    return (
        x.with_columns(
            token=vocab(f"{label}_{c}"),
            token_quantile=np.where(
                np.isfinite(v),
                np.digitize(v, bins=vocab.get_aux(f"{label}_{c}")),
                vocab("nan"),
            ),
        )
        .drop_nulls("token")
        .with_columns(
            tokens=pl.concat_list("token", "token_quantile"),
            times=pl.concat_list("event_time", "event_time"),
        )
    )


def process_cat_val_frame(df: Frame, label: str, vocab: Vocabulary) -> Frame:
    """handle tables that can mostly be described in terms of categories and
    values"""
    return pl.concat(
        process_single_category(x, label, vocab)
        for k, x in df.partition_by("category", as_dict=True).items()
    )


def load_tables(
    hm: pathlib.Path,
) -> dict[str, Frame]:
    """lazy-load all parquet tables from the directory `hm`"""
    return {
        (
            p.stem.split("_")[1] if "assessments" not in p.stem else "assessments"
        ): pl.scan_parquet(p)
        for p in hm.expanduser().glob("*.parquet")
    }


def process_tables(tbl: dict[str, Frame], vocab: Vocabulary) -> dict[str, Frame]:

    tbl["patient"] = (
        tbl["patient"]
        .select("patient_id", "race_category", "ethnicity_category", "sex_category")
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

    tbl["hospitalization"] = (
        tbl["hospitalization"]
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
    if not vocab.has_aux(c):
        vocab.set_aux(c, np.nanquantile(v, np.arange(0.1, 1.0, 0.1)))
    tbl["hospitalization"] = (
        tbl["hospitalization"]
        .with_columns(
            age_at_admission=np.where(
                np.isfinite(v),
                np.digitize(v, bins=vocab.get_aux(c)),
                vocab("nan"),
            )
        )
        .with_columns(
            admission_tokens=pl.concat_list("age_at_admission", "admission_type_name"),
        )
        .drop("age_at_admission", "admission_type_name")
    )

    tbl["adt"] = (
        tbl["adt"]
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

    tbl["labs"] = (
        tbl["labs"]
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
    # TODO: ensure concurrent labs are presented in the same order each time
    tbl["labs"] = process_cat_val_frame(tbl["labs"], label="LAB", vocab=vocab)

    tbl["vitals"] = (
        tbl["vitals"]
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
    tbl["vitals"] = process_cat_val_frame(tbl["vitals"], label="VTL", vocab=vocab)

    tbl["medication"] = (
        tbl["medication"]
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
    tbl["medication"] = process_cat_val_frame(
        tbl["medication"], label="MED", vocab=vocab
    )

    # seems like there's a column for assessment, and then either a
    # numerical_value OR a categorical_value, depending on the assessment
    tbl["assessments"] = (
        tbl["assessments"]
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
    asmt_num = tbl["assessments"].filter(~pl.col("value").is_null())
    asmt_num = process_cat_val_frame(asmt_num, label="ASMT", vocab=vocab).select(
        "hospitalization_id", "event_time", "tokens", "times"
    )

    asmt_cat = (
        tbl["assessments"]
        .filter(pl.col("value").is_null())
        .filter(~pl.col("categorical_value").is_null())
        .with_columns(
            pl.col("category").map_elements(
                vocab, return_dtype=pl.Int64, skip_nulls=False
            ),
            pl.col("categorical_value").map_elements(
                vocab, return_dtype=pl.Int64, skip_nulls=False
            ),
        )
        .with_columns(
            tokens=pl.concat_list("category", "categorical_value"),
            times=pl.concat_list("event_time", "event_time"),
        )
        .select("hospitalization_id", "event_time", "tokens", "times")
    )

    tbl["assessments"] = pl.concat((asmt_num, asmt_cat))

    tbl["respiratory"] = (
        tbl["respiratory"]
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

    # include a token for prone position; this is relatively rare
    tbl["position"] = (
        tbl["position"]
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
        .cast({"times": pl.List(pl.Datetime(time_unit="ms"))})
    )

    return tbl


def get_admission_frame(tbl: dict[str, Frame]) -> Frame:

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
            adm_tokens=pl.concat_list(pl.col("tokens"), pl.col("admission_tokens")),
            adm_times=pl.concat_list(*[pl.col("event_start")] * 5),
        )
        .select(
            "hospitalization_id",
            pl.col("event_start").alias("event_time"),
            "adm_tokens",
            "adm_times",
        )
    )

    return admission_tokens


def get_discharge_frame(tbl: dict[str, Frame]) -> Frame:
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
                lambda x: [x],
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

    return discharge_tokens


def get_events_frame(tbl: dict[str, Frame]) -> Frame:
    events = pl.concat(
        tbl[k].select("hospitalization_id", "event_time", "tokens", "times")
        for k in tbl.keys()
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


def get_tokens_timelines(
    hm: pathlib.Path, vocab: Vocabulary, return_tbl: bool = False
) -> tuple[Frame, dict] | Frame:
    tbl = process_tables(load_tables(hm=hm), vocab)
    adm = get_admission_frame(tbl)
    evt = get_events_frame(tbl)
    dis = get_discharge_frame(tbl)

    # combine the admission tokens, event tokens, and discharge tokens
    tokens_timelines = (
        adm.lazy()
        .join(evt, on="hospitalization_id")
        .join(dis.lazy(), on="hospitalization_id")
        .with_columns(
            tokens=pl.concat_list("adm_tokens", "tokens", "dis_tokens"),
            times=pl.concat_list("adm_times", "times", "dis_times"),
        )
        .select("hospitalization_id", "tokens", "times")
        .collect()
    )

    if return_tbl:
        return tokens_timelines, tbl
    else:
        return tokens_timelines


if __name__ == "__main__":

    if os.uname().nodename.startswith("cri"):
        hm = pathlib.Path("/gpfs/data/bbj-lab/users/burkh4rt/clif-development-sample")
    else:
        # change following line to develop locally
        hm = pathlib.Path("~/Documents/chicago/CLIF/clif-development-sample")

    out_dir = hm.parent.joinpath(hm.stem + "-tokenized").expanduser()
    out_dir.mkdir(exist_ok=True)

    vocab = Vocabulary(tuple(map(lambda i: f"Q{i}", range(10))))
    tokens_timelines, tbl = get_tokens_timelines(hm=hm, vocab=vocab, return_tbl=True)

    vocab.save(out_dir.joinpath("vocab.gzip"))
    tokens_timelines.write_parquet(out_dir.joinpath("tokens_timelines.parquet"))

    """create summary plots
    """

    ct_by_tbl = tbl["hospitalization"].select("hospitalization_id")

    for k in tbl.keys():
        if k not in ("patient", "hospitalization"):
            ct = (
                tbl["hospitalization"]
                .select("hospitalization_id")
                .join(tbl[k], on="hospitalization_id", how="left")
                .group_by("hospitalization_id")
                .agg(pl.col("tokens").count().alias(f"{k}"))
            )
            ct_by_tbl = ct_by_tbl.join(ct, on="hospitalization_id")

    fig = go.Figure()
    for c in ct_by_tbl.columns[1:]:
        fig.add_trace(
            go.Histogram(
                x=ct_by_tbl.select(c).to_numpy().ravel(),
                name=c,
                xbins=dict(start=0, end=400, size=10),
            )
        )

    # Overlay both histograms
    fig.update_layout(barmode="overlay")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()

    fig = px.histogram(
        ct_by_tbl.with_columns(
            pl.sum_horizontal(pl.exclude("hospitalization_id").alias("total"))
        ),
        x="total",
        title="Histogram of total timeline lengths (Log-scale)",
    )
    fig.update_xaxes(range=(0, 10000))
    fig.update_yaxes(type="log")
    fig.show()

    fig = px.histogram(
        ct_by_tbl.with_columns(
            pl.sum_horizontal(pl.exclude("hospitalization_id").alias("total"))
        ),
        x="total",
        title="Histogram of total timeline lengths (Linear-scale)",
    )
    fig.update_xaxes(range=(0, 10000))
    fig.show()

    """ What are some of the most common tokens?
    """

    with pl.Config(tbl_rows=len(vocab)):
        print(
            tokens_timelines.select("tokens")
            .explode("tokens")
            .rename({"tokens": "token"})
            .join(vocab.get_frame(), on="token")
            .select("word")
            .to_series()
            .value_counts()
            .sort("count", descending=True)
        )

    # tbl = load_tables(hm=hm)
