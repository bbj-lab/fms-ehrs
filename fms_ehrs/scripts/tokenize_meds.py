#!/usr/bin/env python3

"""
test scalability of our tokenizer to the full-sized MIMIC version of MEDS
"""

import argparse
import pathlib

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.storage import set_perms
from fms_ehrs.framework.tokenizer import Tokenizer21
from fms_ehrs.framework.tokenizer_base import summarize

logger = get_logger()
logger.info("running {}".format(__file__))
logger.log_env()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=pathlib.Path)
parser.add_argument("--config_loc", type=pathlib.Path)
args, unknowns = parser.parse_known_args()

for k, v in vars(args).items():
    logger.info(f"{k}: {v}")

data_dir, config_loc = map(
    lambda d: pathlib.Path(d).expanduser().resolve(), (args.data_dir, args.config_loc)
)

tkzr = Tokenizer21(data_dir=data_dir, config_file=config_loc)

tokens_timelines = tkzr.get_tokens_timelines()
summarize(tkzr, tokens_timelines, logger=logger)
tokens_timelines = tkzr.pad_and_truncate(tokens_timelines)
set_perms(tokens_timelines.write_parquet)(data_dir / "tokens_timelines.parquet")
tkzr.vocab.save(data_dir / "vocab.gzip")
