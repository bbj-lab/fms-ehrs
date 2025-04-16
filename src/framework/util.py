#!/usr/bin/env python3

"""
utility functions
"""

import logging

import numpy as np
import pandas as pd
import sklearn.metrics as skl_mets
import torch as t


def mvg_avg(x: np.array, w: int = 4) -> np.array:
    """
    moving average for flat array `x` with window size `w`;
    returns array of same length as x
    """
    assert w >= 1
    x_aug = np.concatenate(([x[0]] * (w - 1), x))
    return np.lib.stride_tricks.sliding_window_view(x_aug, w).mean(axis=-1)


def rt_padding_to_left(
    t_rt_pdd: t.Tensor, pd_tk: int, unif_rand_trunc: bool = False
) -> t.Tensor:
    """
    take a tensor `t_rt_pdd` padded on the right with padding token `pd_tk` and
    move that padding to the left; if `unif_rand_trunc`, truncate sequence
    uniformly at random
    """
    i = t.argmax(
        (t_rt_pdd == pd_tk).int()
    ).item()  # either the index of the first padding token or 0
    if unif_rand_trunc and i > 0:
        i = t.randint(
            low=1, high=i, size=(1,)
        ).item()  # new cutpoint chosen uniformly at random from seq length
    return (
        t.concat([t.full((t_rt_pdd.shape[0] - i,), pd_tk), t_rt_pdd[:i]])
        if i > 0
        else t_rt_pdd  # if no padding was present
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


def ragged_lists_to_array(ls_arr: list[np.array]) -> np.array:
    """
    form an 2d-array from a collection of variably-sized 1d-arrays
    """
    n, m = len(ls_arr), max(map(len, ls_arr))
    arr = np.full(shape=(n, m), fill_value=np.nan, dtype=type(next(iter(ls_arr))[0]))
    for i, x in enumerate(ls_arr):
        arr[i, : len(x)] = x
    return arr


def set_pd_options():
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.width = 250
    pd.options.display.max_colwidth = 100
