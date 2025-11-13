#!/usr/bin/env python3

"""
partition groups by order of appearance in the dataset into train-validation-test
sets at the subject level; alternately, extract a small sample for development purposes
"""

import argparse
import io
import itertools
import pathlib

import polars as pl
import ruamel.yaml as yaml

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_version_out", type=str, default="test")
parser.add_argument("--data_dir_out", type=pathlib.Path, default="../../tmp-test/")
parser.add_argument("--data_dir_in", type=pathlib.Path, default="../../tmp-test/raw")
parser.add_argument("--train_frac", type=float, default=0.7)
parser.add_argument("--val_frac", type=float, default=0.1)
parser.add_argument("--valid_admission_window", nargs=2, type=str)
parser.add_argument("--match_other_split", type=pathlib.Path)
parser.add_argument(
    "--config_loc", type=pathlib.Path, default="../fms_ehrs/config/config-20.yaml"
)
parser.add_argument("--development_sample", action="store_true")
parser.add_argument("--dev_frac", type=float, default=0.01)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

config = (
    yaml.YAML(typ="safe").load(pathlib.Path(args.config_loc).expanduser().resolve())
    if args.config_loc is not None
    else dict()
)

logger.info("begin config ---")
with io.StringIO() as s:
    yaml.YAML(typ="safe", pure=True).dump(config, s)
    for line in s.getvalue().splitlines():
        logger.info(line)
logger.info("... end config")

config_ref = config.get("reference", {})
ref_tbl_str = config_ref.get("table", "clif_hospitalization")
sbj_id_str = config.get("subject_id", "hospitalization_id")
grp_id_str = config.get("group_id", "patient_id")

data_dir_in, data_dir_out = map(
    lambda d: pathlib.Path(d).expanduser().resolve(),
    (args.data_dir_in, args.data_dir_out),
)

# make output sub-directories
splits = ("train", "val", "test") if not args.development_sample else ("dev",)
dirs_out = dict()
for s in splits:
    dirs_out[s] = data_dir_out.joinpath(args.data_version_out, s)
    dirs_out[s].mkdir(exist_ok=True, parents=True)

sbj_ids = dict()
grp_ids = dict()
aug_tbls = [t["table"] for t in config.get("augmentation_tables", {}) if "table" in t]

ref = (
    pl.scan_parquet(data_dir_in.joinpath("{}.parquet".format(ref_tbl_str)))
    .filter(pl.col(config_ref.get("age", "age_at_admission")) >= 18)
    .cast(
        {
            config_ref.get("start_time", "admission_dttm"): pl.Datetime(time_unit="ms"),
            grp_id_str: pl.String,
            sbj_id_str: pl.String,
        }
    )
    .filter(
        pl.col(config_ref.get("start_time", "admission_dttm")).is_between(
            pl.lit(args.valid_admission_window[0]).cast(pl.Date),
            pl.lit(args.valid_admission_window[1]).cast(pl.Date),
        )
        if args.valid_admission_window is not None
        else True
    )
)

# partition patient ids
group_ids = (
    ref.group_by(grp_id_str)
    .agg(
        pl.col(config_ref.get("start_time", "admission_dttm")).min().alias("first_time")
    )
    .sort("first_time")
    .select(grp_id_str)
    .collect()
)

n_total = group_ids.n_unique()

if args.development_sample:  # make a development sample
    n_dev = int(args.dev_frac * n_total)
    logger.info(f"{grp_id_str} {n_total=}")
    logger.info(f"Partition: {n_dev=}")
    grp_ids["dev"] = group_ids.head(n_dev)
else:  # regular split
    n_train = int(args.train_frac * n_total)
    n_val = int(args.val_frac * n_total)
    if (n_test := n_total - n_train - n_val) < 0:
        raise f"check {args.train_frac=} and {args.val_frac=}"
    logger.info(f"{grp_id_str} {n_total=}")
    logger.info(f"Partition: {n_train=}, {n_val=}, {n_test=}")
    grp_ids["train"] = group_ids.head(n_train)
    grp_ids["val"] = group_ids.slice(n_train, n_val)
    grp_ids["test"] = group_ids.tail(n_test)
    assert sum(list(map(lambda x: x.n_unique(), grp_ids.values()))) == n_total

for s0, s1 in itertools.combinations(splits, 2):
    assert grp_ids[s0].join(grp_ids[s1], on=grp_id_str).n_unique() == 0

# partition subjects according to the group split
subject_ids = ref.select(grp_id_str, sbj_id_str).unique().collect()

for s in splits:
    sbj_ids[s] = subject_ids.join(grp_ids[s], on=grp_id_str).select(sbj_id_str)

for s0, s1 in itertools.combinations(splits, 2):
    assert sbj_ids[s0].join(sbj_ids[s1], on=sbj_id_str).n_unique() == 0

n_total = sum(list(map(lambda x: x.n_unique(), sbj_ids.values())))
assert n_total <= subject_ids.n_unique()

logger.info(f"{sbj_id_str} {n_total=}")
if not args.development_sample:
    logger.info(
        f"Partition: {sbj_ids['train'].n_unique()=}, "
        f"{sbj_ids['val'].n_unique()=}, {sbj_ids['test'].n_unique()=}"
    )

# generate sub-tables
for s in splits:
    set_perms(
        pl.scan_parquet(data_dir_in.joinpath("{}.parquet".format(ref_tbl_str)))
        .with_columns(pl.col(sbj_id_str).cast(pl.String))
        .join(sbj_ids[s].lazy(), on=sbj_id_str)
        .sink_parquet
    )(dirs_out[s].joinpath("{}.parquet".format(ref_tbl_str)))
    for t in data_dir_in.glob("*.parquet"):
        if t.stem != ref_tbl_str:
            try:
                set_perms(
                    pl.scan_parquet(t)
                    .with_columns(pl.col(sbj_id_str).cast(pl.String))
                    .join(sbj_ids[s].lazy(), on=sbj_id_str)
                    .sink_parquet
                )(dirs_out[s].joinpath(t.name))
                logger.info(f"Created {t.name} in {s} with {sbj_id_str}")
                continue
            except pl.exceptions.ColumnNotFoundError:
                logger.warning(f"Failed to find {sbj_id_str} in {t.name}")
            try:
                set_perms(
                    pl.scan_parquet(t)
                    .with_columns(pl.col(grp_id_str).cast(pl.String))
                    .join(grp_ids[s].lazy(), on=grp_id_str)
                    .sink_parquet
                )(dirs_out[s].joinpath(t.name))
                logger.info(f"Created {t.name} in {s} with {grp_id_str}")
                continue
            except pl.exceptions.ColumnNotFoundError:
                logger.warning(f"Failed to find {grp_id_str} in {t.name}")

logger.info("---fin")
