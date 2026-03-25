#!/usr/bin/env python3

"""
Run the Epi-of-Sedation project code to flag successful extubations
cf. https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-epi-of-sedation
"""

import argparse
import pathlib

import clifpy as cpy

# import requests
import duckdb as ddb
import pandas as pd
import polars as pl

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=pathlib.Path, default="../../data-raw/mimic-2.1.0"
)
parser.add_argument("--q_dir", type=pathlib.Path, default="../fms_ehrs/misc")
parser.add_argument("--tz", type=str, default="UTC")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, q_dir = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.q_dir)
)
opts = {"data_directory": str(data_dir), "timezone": args.tz, "filetype": "parquet"}

pt_df = cpy.Patient.from_file(**opts, columns=["patient_id", "death_dttm"]).df

hosp_df = cpy.Hospitalization.from_file(
    **opts,
    columns=[
        "patient_id",
        "hospitalization_id",
        "discharge_dttm",
        "discharge_category",
        "age_at_admission",
    ],
).df

vitals_df = cpy.Vitals.from_file(**opts).df
last_vitals_df = (
    vitals_df.groupby("hospitalization_id").agg({"recorded_dttm": "max"}).reset_index()
)

resp_p = pl.read_parquet(
    data_dir.joinpath("clif_respiratory_support_processed.parquet")
).to_pandas()

all_streaks = ddb.sql(q_dir.joinpath("modified_cohort_id.sql").read_text())

q = """
FROM all_streaks
SELECT hospitalization_id, _streak_id, _start_dttm, _end_dttm, _duration_hrs
WHERE _on_imv = 1 -- has to be an IMV streak
    AND _streak_id = 1 -- has to be the first IMV streak
"""
cohort_imv_streaks_f = ddb.sql(q).df()

q = """
FROM hosp_df
INNER JOIN cohort_imv_streaks_f USING (hospitalization_id)
SELECT DISTINCT patient_id, hospitalization_id
"""
pt_to_hosp_id_mapper = ddb.sql(q).df()
cohort_pt_ids = pt_to_hosp_id_mapper["patient_id"].tolist()

cs_df = cpy.CodeStatus.from_file(
    **opts,
    columns=["patient_id", "start_dttm", "code_status_category"],
    filters={"patient_id": cohort_pt_ids},
).df

q = """
FROM cs_df
LEFT JOIN pt_to_hosp_id_mapper USING (patient_id)
SELECT hospitalization_id, start_dttm, code_status_category
"""
cs_df = ddb.sql(q).df()

con = ddb.connect()
con.register("last_vitals_df", last_vitals_df)
con.register("resp_p", resp_p)
con.register("cs_df", cs_df)
con.register("hosp_df", hosp_df)
sbt_outcomes = con.sql(q_dir.joinpath("modified_whiskey.sql").read_text()).df()

# whiskey_sql_url = (
#     "https://raw.githubusercontent.com/Common-Longitudinal-ICU-data-Format"
#     + "/CLIF-epi-of-sedation/c75107d0738dd75db0e6b81789ad8ef3c0d1f5f6"
#     + "/code/sbt.sql"
# )
# sbt_outcomes = con.sql(requests.get(whiskey_sql_url).text).df()

extub_flag = sbt_outcomes[["hospitalization_id", "event_dttm", "_success_extub"]].loc[
    lambda x: x._success_extub == 1
]
extub_flag.event_dttm += pd.Timedelta(
    hours=6
)  # not actually a success until the 6 hours pass

logger.info("Found...")
logger.info("{ni} intubations".format(ni=sum(sbt_outcomes._intub)))
logger.info("Resulting in...")
logger.info("{nfe} first extubations".format(nfe=sum(sbt_outcomes._extub_1st)))
logger.info("    {ne} successful".format(ne=len(extub_flag)))
logger.info(
    "    {nwl} associated with reintubation w/in 6hrs".format(
        nwl=sum(sbt_outcomes._fail_extub)
    )
)
logger.info(
    "    {nwl} associated with withdrawal of life support".format(
        nwl=sum(sbt_outcomes._withdrawl_lst)
    )
)
logger.info(
    "    {nwl} associated with death/hospice".format(
        nwl=sum(sbt_outcomes._death_after_extub_wo_reintub)
    )
)

set_perms(extub_flag.to_parquet)(
    out_loc := data_dir.joinpath("clif_successful_extubation.parquet")
)
logger.info(f"Saved frame to {out_loc}.")


logger.info("---fin")
