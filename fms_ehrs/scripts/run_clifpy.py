#!/usr/bin/env python3

"""
Run the CLIFpy package on CLIF-2.1 datasets
"""

import argparse
import pathlib

import clifpy as cp
from clifpy.tables.respiratory_support import RespiratorySupport

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=pathlib.Path, default="../../data-raw/mimic-2.1.0"
)
parser.add_argument("--out_dir", type=pathlib.Path, default="../../figs")
parser.add_argument("--tz", type=str, default="US/Eastern")
parser.add_argument(
    "--tables",
    type=str,
    nargs="*",
    default=[
        "patient",
        "hospitalization",
        "adt",
        "labs",
        "vitals",
        "medication_admin_continuous",
        "medication_admin_intermittent",
        "patient_assessments",
        "respiratory_support",
        "patient_procedures",
        "code_status",
    ],
)
parser.add_argument("--validate", action="store_true")
parser.add_argument("--waterfall", action="store_true")
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, out_dir = map(
    lambda x: pathlib.Path(x).expanduser().resolve(), (args.data_dir, args.out_dir)
)

if args.validate:
    orchestrator = cp.ClifOrchestrator(
        data_directory=str(data_dir),
        filetype="parquet",
        timezone=args.tz,
        output_directory=str(out_dir),
    )
    orchestrator.initialize(tables=args.tables)
    orchestrator.validate_all()

if args.waterfall:
    resp_support = RespiratorySupport.from_file(
        data_directory=str(data_dir), filetype="parquet", timezone=args.tz
    )
    processed = resp_support.waterfall()
    processed.validate()
    set_perms(processed.df.to_parquet)(
        data_dir.joinpath("clif_respiratory_support_processed.parquet")
    )


logger.info("---fin")
