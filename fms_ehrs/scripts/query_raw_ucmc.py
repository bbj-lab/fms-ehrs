#!/usr/bin/env python3

"""
look up diagnostic data for provided hospitalization id's in UCMC
"""

import argparse
import pathlib

import polars as pl

from fms_ehrs.framework.logger import get_logger

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ucmc_dir",
    type=pathlib.Path,
    default="../../ucmc",
    help="""The directory containing the pipe-delimited diagnoses file.""",
)
parser.add_argument(
    "--har_ids",
    type=str,
    nargs="*",
    default=["8797520", "27055120", "10969205", "2974992", "20528107"],
)


args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

ucmc_dir = pathlib.Path(args.ucmc_dir).expanduser().resolve()
dx = pl.read_csv(ucmc_dir / "C19_DX_DID.txt", separator="|", infer_schema=False)

for har_id in args.har_ids:
    print(dx.filter(pl.col("C19_PATIENT_ID") == har_id))


logger.info("---fin")
