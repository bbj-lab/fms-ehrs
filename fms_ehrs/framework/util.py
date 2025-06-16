#!/usr/bin/env python3

"""
utility functions
"""

import collections
import copy
import logging
import os
import pathlib
import typing

import numpy as np
import pandas as pd
import torch as t

from fms_ehrs.framework.logger import get_logger
from fms_ehrs.framework.vocabulary import Vocabulary

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike
Dictlike: typing.TypeAlias = collections.OrderedDict | dict


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
        ).item()  # new cut-point chosen uniformly at random from seq length
    return (
        t.concat([t.full((t_rt_pdd.shape[0] - i,), pd_tk), t_rt_pdd[:i]])
        if i > 0
        else t_rt_pdd  # if no padding was present
    )


def ragged_lists_to_array(ls_arr: typing.List[np.array]) -> np.array:
    """
    form an 2d-array from a collection of variably-sized 1d-arrays
    """
    n, m = len(ls_arr), max(map(len, ls_arr))
    arr = np.full(shape=(n, m), fill_value=np.nan)
    for i, x in enumerate(ls_arr):
        arr[i, : len(x)] = x
    return arr


def extract_examples(
    timelines: np.array,
    criteria: np.array,
    vocab: Vocabulary,
    flags: list = None,
    k: int = 10,
    w_sz: int = 3,
    lag: int = 0,
    logger: logging.Logger = get_logger(),
    top_k: bool = True,
) -> None:
    assert timelines.shape[0] == criteria.shape[0]
    assert timelines.shape[1] == criteria.shape[1] + lag
    if flags:
        assert len(flags) == timelines.shape[0]
    top_k_flat_idx = (
        np.argsort(np.nan_to_num(criteria.flatten()))[::-1][:k]
        if top_k
        else np.argsort(np.nan_to_num(criteria.flatten(), nan=np.inf))[:k]  # bottom k
    )
    top_k_idx = np.array(np.unravel_index(top_k_flat_idx, criteria.shape)).T
    m = timelines.shape[-1]
    for i0, i1 in top_k_idx:
        ints = timelines[i0, max(0, i1 - w_sz) : min(m - 1, i1 + w_sz + lag)]
        tkns = "->".join(
            s if (s := vocab.reverse[i]) is not None else "None" for i in ints
        )
        hit = " ".join(
            s if (s := vocab.reverse[i]) is not None else "None"
            for i in timelines[i0][i1 : i1 + lag + 1]
        )
        if flags:
            logger.info(f"{i0=}, {i1=} | {flags[i0]}")
        else:
            logger.info(f"{i0=}, {i1=} ")
        logger.info(f"{hit=} in {tkns}")
        logger.info(
            "->".join(
                map(
                    str,
                    criteria[i0, max(0, i1 - w_sz) : min(m - 1, i1 + w_sz + lag)].round(
                        2
                    ),
                )
            )
        )


def redact_tokens_times(
    tks_arr: typing.List[np.array],
    tms_arr: typing.List[np.array],
    inf_arr: np.array,
    *,
    k: int = None,
    pct: float = None,
    method: typing.Literal["top", "bottom", "random"] = "top",
    aggregation: typing.Literal["max", "sum"] = "max",
    rng: np.random._generator.Generator = np.random.default_rng(seed=42),
) -> tuple[np.array, np.array]:
    """given an array `tks_arr` of arrays of tokens and an array `tms_arr` of
    arrays of times, and an array `inf_arr` containing the information content
    up to a certain cutoff of the tokens in each timeline, iterate through the
    timelines and drop all tokens corresponding to events containing the (`top`)
    most informative, (`bottom`) least informative, or (`random`) randomly
    chosen tokens (not including the prefix, which we always keep); we specify
    the number of events either as fixed `k` for all timelines or as a `pct` of
    the total number of events in each timeline; one and only one of these should
    be specified
    """
    assert len(tks_arr) == len(tms_arr) == len(inf_arr)
    assert (k is not None) ^ (pct is not None)  # xor
    tks_new = copy.deepcopy(tks_arr)
    tms_new = copy.deepcopy(tms_arr)
    for i in range(len(tks_new)):
        tks, tms = tks_arr[i], tms_arr[i]
        tlen = min(len(tks), len(tms))
        tks, tms = tks[:tlen], tms[:tlen]
        tms_unq, idx = np.unique(tms, return_inverse=True)
        if method in ("top", "bottom"):
            infm = inf_arr[i, :tlen]
            if aggregation == "max":
                result = np.full(tms_unq.shape, -np.inf)
                np.maximum.at(result, idx, infm)
            elif aggregation == "sum":
                result = np.zeros(shape=tms_unq.shape)
                np.add.at(result, idx, infm)
            else:
                raise Exception(f"Check {aggregation=}")
            srt = np.argsort(result)
            if method == "top":
                srt = srt[::-1]
        elif method == "random":
            srt = rng.permutation(len(tms_unq))
        else:
            raise Exception(f"Check {method=}")
        srt = srt[srt != idx[0]]  # don't drop prefix
        if k is not None:
            to_drop = srt[:k]
        elif pct is not None:
            to_drop = srt[: int(pct * len(srt))]
        tks_new[i] = tks[~np.isin(idx, to_drop)]
        tms_new[i] = tms[~np.isin(idx, to_drop)]
    return tks_new, tms_new


def set_pd_options() -> None:
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.width = 250
    pd.options.display.max_colwidth = 100


if __name__ == "__main__":

    print(ragged_lists_to_array([[2.0, 3.0], [3.0]]))

    tks = [np.arange(10)]
    tms = [np.array([0] * 3 + [1] * 3 + [2] * 3 + [3])]
    inf = np.array([0] * 3 + [3, 0, 0] + [2] * 3 + [1]).reshape(1, -1)
    print(redact_tokens_times(tks, tms, inf, k=1))
    print(redact_tokens_times(tks, tms, inf, k=1, aggregation="sum"))
    print(redact_tokens_times(tks, tms, inf, k=1, method="random"))

    tms_unq, idx = np.unique(tms, return_inverse=True)
    result = np.zeros(shape=tms_unq.shape)
    np.add.at(result, idx, inf.ravel())
    print(result[idx])
