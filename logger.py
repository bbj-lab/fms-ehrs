#!/usr/bin/env python3

""" add some context to stddout files
"""

import datetime
import logging
import os
import subprocess
import sys


class SlurmLogger(logging.Logger):
    def __init__(self, name: str = "clif-tokenizer"):
        super().__init__(name=name)
        self.setLevel(logging.INFO)
        self.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", "%Y-%m-%dT%H:%M:%S%z"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.addHandler(ch)
        self.propagate = False

    def log_env(self):

        self.info("running Python {}".format(sys.version))
        self.info("on {}".format(os.uname().nodename))
        self.info("tz-info: {}".format(datetime.datetime.now().astimezone().tzinfo))
        if slurm_job_id := os.getenv("SLURM_JOB_ID", ""):
            self.info("slurm job id: {}".format(slurm_job_id))

        smi = subprocess.run(
            "nvidia-smi -L",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
        if smi.returncode == 0:
            for gpu_i in smi.stdout.decode().strip().split("\n"):
                self.info(gpu_i)

        get_git = subprocess.run(
            "git rev-parse --short HEAD",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
        if get_git.returncode == 0:
            self.info("commit: {}".format(get_git.stdout.decode().strip().upper()))

        get_branch = subprocess.run(
            "git rev-parse --abbrev-ref HEAD",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
        if get_branch.returncode == 0:
            self.info("branch: {}".format(get_branch.stdout.decode().strip().upper()))


def get_logger() -> SlurmLogger:
    logging.setLoggerClass(SlurmLogger)
    logger = logging.getLogger("clif-tokenizer")
    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.log_env()
