#!/usr/bin/env python3

"""
determine some outcomes of interest for each hospitalization
"""

import argparse
import pathlib

import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.vocabulary import Vocabulary

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path, default="../../data-mimic")
parser.add_argument("--ref_version", type=str, default="X21")
parser.add_argument("--data_version", type=str, default="X21_first_24h")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")


data_dir = pathlib.Path(args.data_dir).expanduser().resolve()

# load and prep data
splits = ("train", "val", "test")
data_dirs = dict()
ref_dirs = dict()
for s in splits:
    data_dirs[s] = data_dir.joinpath(f"{args.data_version}-tokenized", s)
    ref_dirs[s] = data_dir.joinpath(f"{args.ref_version}-tokenized", s)

vocab = Vocabulary().load(ref_dirs["train"].joinpath("vocab.gzip"))
expired_token = vocab("DSCG_Expired")
icu_token = vocab("XFR-IN_icu")
imv_token = vocab("RESP_imv")
ama_token = vocab("DSCG_Against_Medical_Advice_(AMA)")
hosp_token = vocab("DSCG_Hospice")
# sofa_2_plus_tokens = [
#     vocab(s)
#     for sys in ["cv", "cns", "coag", "liver", "renal", "resp"]
#     for num in range(2, 5)
#     if (s := f"SOFA_{sys}-{num}") in vocab.lookup
# ]


for s in splits:
    outcomes = (
        pl.scan_parquet(ref_dirs[s].joinpath("tokens_timelines.parquet"))
        .with_columns(
            length_of_stay=(
                pl.col("times").list.get(-1) - pl.col("times").list.get(0)
            ).dt.total_hours(),
            same_admission_death=pl.col("tokens").list.contains(expired_token),
            icu_admission=pl.col("tokens").list.contains(icu_token),
            imv_event=pl.col("tokens").list.contains(imv_token),
            ama_discharge=pl.col("tokens").list.contains(ama_token),
            hospice_discharge=pl.col("tokens").list.contains(hosp_token),
            # sofa_2_plus=pl.col("tokens")
            # .list.eval(pl.element().is_in(sofa_2_plus_tokens))
            # .list.any(),
        )
        .with_columns(
            long_length_of_stay=pl.col("length_of_stay") > 24 * 7  # 7 days in hours
        )
        .select(
            "hospitalization_id",
            "length_of_stay",
            "same_admission_death",
            "long_length_of_stay",
            "icu_admission",
            "imv_event",
            "ama_discharge",
            "hospice_discharge",
        )
    )
    (
        set_perms(
            pl.scan_parquet(data_dirs[s].joinpath("tokens_timelines.parquet"))
            .with_columns(
                icu_admission_24h=pl.col("tokens").list.contains(icu_token),
                imv_event_24h=pl.col("tokens").list.contains(imv_token),
            )
            .join(
                outcomes,
                how="left",
                on="hospitalization_id",
                validate="1:1",
                maintain_order="left",
            )
            .collect()
            .write_parquet
        )(data_dirs[s].joinpath("tokens_timelines_outcomes.parquet"))
    )

logger.info("---fin")
