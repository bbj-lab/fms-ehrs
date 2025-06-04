#!/usr/bin/env python3

""" add some context to stddout files
"""

import datetime
import functools
import inspect
import logging
import os
import subprocess
import sys

import numpy as np
from sklearn import metrics as skl_mets


class SlurmLogger(logging.Logger):
    def __init__(self, name: str = "fms-ehrs-reps"):
        super().__init__(name=name)
        self.setLevel(logging.INFO)
        self.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", "%Y-%m-%dT%H:%M:%S%z"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        if os.getenv("RANK", "0") == "0":
            self.addHandler(ch)
        self.propagate = False

    def log_env(self):
        self.info("from {}".format(os.getcwd()))
        self.info("with Python {}".format(sys.version))
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
            self.info("commit: {}".format(get_git.stdout.decode().strip()))

        get_branch = subprocess.run(
            "git rev-parse --abbrev-ref HEAD",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
        if get_branch.returncode == 0:
            self.info("branch: {}".format(get_branch.stdout.decode().strip()))

    def log_calls(self, func: callable) -> callable:

        @functools.wraps(func)
        def log_io(*args, **kwargs):
            func_args = inspect.signature(func).bind_partial(*args, **kwargs)
            func_args.apply_defaults()
            self.info(f"{func.__name__} called with---")
            for k, v in func_args.arguments.items():
                self.info(f"{k}: {v}")
            y = func(*args, **kwargs)
            self.info(f"---{func.__name__}")
            return y

        return log_io


def get_logger() -> SlurmLogger | logging.Logger:
    logging.setLoggerClass(SlurmLogger)
    return logging.getLogger("fms-ehrs-reps")


def log_summary(arr: np.array, logger: logging.Logger) -> None:
    """log some summary stats for the array `arr`"""
    logger.info("Array of shape: {}".format(arr.shape))
    logger.info("Pct non-nan: {:.2f}".format(100 * np.isfinite(arr).mean()))
    logger.info("Range: ({:.2f}, {:.2f})".format(np.nanmin(arr), np.nanmax(arr)))
    logger.info("Mean: {:.2f}".format(np.nanmean(arr)))
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999):  # 0.0001, 0.001, 0.01, 0.1,
        logger.info(
            "{:05.2f}% quantile: {:.2f}".format(100 * q, np.nanquantile(arr, q))
        )


def log_classification_metrics(
    y_true: np.array, y_score: np.array, logger: logging.Logger
):
    """evaluate a classifier under a variety of metrics"""
    assert y_true.shape[0] == y_score.shape[0]

    logger.info(
        "roc_auc: {:.3f}".format(skl_mets.roc_auc_score(y_true=y_true, y_score=y_score))
    )

    for met in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
    ):
        logger.info(
            "{}: {:.3f}".format(
                met,
                getattr(skl_mets, f"{met}_score")(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
            )
        )


if __name__ == "__main__":

    from src.framework.stats import generate_classifier_preds

    logger = get_logger()
    logger.log_env()

    y_true, y_preds = generate_classifier_preds(num_preds=1)
    log_classification_metrics(y_true, y_preds[0], logger)
    log_summary(y_preds[0], logger)
