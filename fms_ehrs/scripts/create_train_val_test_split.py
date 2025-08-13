#!/usr/bin/env python3

"""
partition patients by order of appearance in the dataset into train-validation-test sets
"""

import itertools
import os
import pathlib

import fire as fi
import polars as pl

from fms_ehrs.framework.logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()


@logger.log_calls
def main(
    *,
    data_version_out: str = "raw",
    data_dir_in: os.PathLike = None,
    data_dir_out: os.PathLike = None,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    valid_admission_window: tuple[str, str] = None,
):
    data_dir_in, data_dir_out = map(
        lambda d: pathlib.Path(d).expanduser().resolve(), (data_dir_in, data_dir_out)
    )

    # make output sub-directories
    splits = ("train", "val", "test")
    dirs_out = dict()
    for s in splits:
        dirs_out[s] = data_dir_out.joinpath(data_version_out, s)
        dirs_out[s].mkdir(exist_ok=True, parents=True)

    hosp_prepoc = (
        pl.scan_parquet(data_dir_in.joinpath("clif_hospitalization.parquet"))
        .filter(pl.col("age_at_admission") >= 18)
        .cast({"admission_dttm": pl.Datetime(time_unit="ms")})
        .filter(
            pl.col("admission_dttm").is_between(
                pl.lit(valid_admission_window[0]).cast(pl.Date),
                pl.lit(valid_admission_window[1]).cast(pl.Date),
            )
            if valid_admission_window is not None
            else True
        )
    )

    # partition patient ids
    patient_ids = (
        hosp_prepoc.group_by("patient_id")
        .agg(pl.col("admission_dttm").min().alias("first_admission"))
        .sort("first_admission")
        .select("patient_id")
        .collect()
    )

    n_total = patient_ids.n_unique()
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    n_test = n_total - (n_train + n_val)

    logger.info(f"Patients {n_total=}")
    logger.info(f"Partition: {n_train=}, {n_val=}, {n_test=}")

    if n_test < 0:
        raise f"check {train_frac=} and {val_frac=}"

    p_ids = dict()
    p_ids["train"] = patient_ids.head(n_train)
    p_ids["val"] = patient_ids.slice(n_train, n_val)
    p_ids["test"] = patient_ids.tail(n_test)

    for s0, s1 in itertools.combinations(splits, 2):
        assert p_ids[s0].join(p_ids[s1], on="patient_id").n_unique() == 0

    assert sum(list(map(lambda x: x.n_unique(), p_ids.values()))) == n_total

    # partition hospitalization ids according to the patient split
    hospitalization_ids = (
        hosp_prepoc.select("patient_id", "hospitalization_id").unique().collect()
    )

    h_ids = dict()
    for s in splits:
        h_ids[s] = hospitalization_ids.join(p_ids[s], on="patient_id").select(
            "hospitalization_id"
        )

    for s0, s1 in itertools.combinations(splits, 2):
        assert h_ids[s0].join(h_ids[s1], on="hospitalization_id").n_unique() == 0

    n_total = sum(list(map(lambda x: x.n_unique(), h_ids.values())))
    assert n_total <= hospitalization_ids.n_unique()

    logger.info(f"Hospitalizations {n_total=}")
    logger.info(
        f"Partition: {h_ids['train'].n_unique()=}, "
        f"{h_ids['val'].n_unique()=}, {h_ids['test'].n_unique()=}"
    )

    # generate sub-tables
    for s in splits:
        pl.scan_parquet(data_dir_in.joinpath("clif_patient.parquet")).join(
            p_ids[s].lazy(), on="patient_id"
        ).sink_parquet(dirs_out[s].joinpath("clif_patient.parquet"))

        pl.scan_parquet(data_dir_in.joinpath("clif_hospitalization.parquet")).join(
            h_ids[s].lazy(), on="hospitalization_id"
        ).sink_parquet(dirs_out[s].joinpath("clif_hospitalization.parquet"))

        for t in data_dir_in.glob("*.parquet"):
            if t.stem.split("_", 1)[1] not in ("hospitalization", "patient"):
                pl.scan_parquet(t).join(
                    h_ids[s].lazy(), on="hospitalization_id"
                ).sink_parquet(dirs_out[s].joinpath(t.name))


if __name__ == "__main__":
    fi.Fire(main)
