#!/usr/bin/env python3

"""
utility functions
"""

import collections
import copy
import functools
import os
import pathlib
import typing

import numba as nb
import numpy as np
import pandas as pd
import torch as t

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike
Dictlike: typing.TypeAlias = collections.OrderedDict | dict


def mvg_avg(x: np.ndarray, w: int = 4) -> np.ndarray:
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


def ragged_lists_to_array(
    ls_arr: typing.List[np.ndarray] | typing.List[typing.List],
) -> np.ndarray:
    """
    form an 2d-array from a collection of variably-sized 1d-arrays
    """
    n, m = len(ls_arr), max(map(len, ls_arr))
    arr = np.full(shape=(n, m), fill_value=np.nan)
    for i, x in enumerate(ls_arr):
        arr[i, : len(x)] = x
    return arr


def collate_events_info(
    times: np.ndarray,
    info: np.ndarray,
    aggregation: typing.Literal["max", "sum"] = "sum",
):
    """given an array of `tokens` that occur at `times` and have context-
    aware information `info`, groups these tokens into events and calculates
    information for these events according to the given `aggregation`
    """
    assert times.size == info.size
    times_uniq, times_idx = np.unique(times, return_inverse=True)
    if aggregation == "max":
        info_agg = np.full(times_uniq.shape, -np.inf)
        np.maximum.at(info_agg, times_idx, info)
    elif aggregation == "sum":
        info_agg = np.zeros(shape=times_uniq.shape)
        np.add.at(info_agg, times_idx, info)
    else:
        raise Exception("Check aggregation.")
    return info_agg, times_idx


def redact_tokens_times(
    tks_arr: typing.List[np.ndarray],
    tms_arr: typing.List[np.ndarray],
    met_arr: np.ndarray,
    *,
    k: int = None,
    pct: float = None,
    method: typing.Literal["top", "bottom", "random"] = "top",
    aggregation: typing.Literal["max", "sum"] = "sum",
    rng: np.random._generator.Generator = np.random.default_rng(seed=42),
) -> tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """given an array of tokens happening at the corresponding array `tms_arr` of
    of times, and an array `met_arr` containing some metric on tokens
    up to a certain cutoff of the tokens in each timeline, iterate through the
    timelines and drop all tokens corresponding to events containing the (`top`)
    most informative, (`bottom`) least informative, or (`random`) randomly
    chosen events (not including the prefix, which we always keep); we specify
    the number of events either as fixed `k` for all timelines or as a `pct` of
    the total number of events in each timeline; one and only one of these should
    be specified
    """
    assert len(tks_arr) == len(tms_arr) == len(met_arr)
    assert (k is not None) ^ (pct is not None)  # xor
    tks_new = copy.deepcopy(tks_arr)
    tms_new = copy.deepcopy(tms_arr)
    for i in range(len(tks_new)):
        tks, tms = tks_arr[i], tms_arr[i]
        tlen = min(len(tks), len(tms))
        tks, tms, infm = tks[:tlen], tms[:tlen], met_arr[i, :tlen]
        if method in ("top", "bottom"):
            result, idx = collate_events_info(tms, infm, aggregation)
            srt = np.argsort(result)
            if method == "top":
                srt = srt[::-1]
        elif method == "random":
            tms_unq, idx = np.unique(tms, return_inverse=True)
            srt = rng.permutation(len(tms_unq))
        else:
            raise Exception(f"Check {method=}")
        srt = srt[srt != idx[0]]  # don't drop prefix
        to_drop = srt[:k] if k is not None else srt[: int(pct * len(srt))]
        tks_new[i] = tks[~np.isin(idx, to_drop)]
        tms_new[i] = tms[~np.isin(idx, to_drop)]
    return tks_new, tms_new


def mean_log(arr: np.ndarray, **kwargs) -> np.ndarray:
    return np.nanmean(np.log(arr), **kwargs)


def median_log(arr: np.ndarray, **kwargs) -> np.ndarray:
    return np.nanmedian(np.log(arr), **kwargs)


@functools.cache
def lookahead(n: int, w: int) -> np.ndarray:
    """return a mask with lookahead window `w` on an n n matrix"""
    return np.tri(n) - np.tri(n, k=-w)


def agg_str2fn(agg_fn_str: str) -> typing.Callable:
    """turn string representation of aggregation function into a function"""
    match agg_fn_str:
        case "mean_log":
            return mean_log
        case "median_log":
            return median_log
        case _ if agg_fn_str.startswith("Q"):
            return functools.partial(np.quantile, q=int(agg_fn_str[1:]) / 100)
        case _:
            return getattr(np, agg_fn_str)


def attention_rollout(attentions: np.ndarray) -> np.ndarray:
    """
    take `attentions` (n_layers × batch_size × num_heads × sequence_length
    × sequence_length) and compute rollout-based importance
    """
    return functools.reduce(
        lambda x, y: y @ x,
        list(
            0.5
            * (
                attentions
                + np.tile(
                    np.eye(attentions.shape[-1]), reps=(*attentions.shape[:3], 1, 1)
                )
            )
        ),
    )


@nb.jit(nb.float32[:, :, :, :](nb.float32[:, :, :, :, :]), nopython=True)
def attention_rollout_numba(attentions: np.ndarray) -> np.ndarray:
    I_n = np.zeros(shape=attentions[0].shape, dtype=np.float32)
    _I = np.eye(attentions.shape[-1], dtype=np.float32)
    for i in nb.prange(attentions[0].shape[0]):
        for j in range(attentions[0].shape[1]):
            I_n[i, j] = _I
    ret = np.float32(0.5) * (attentions[0] + I_n)
    for i in range(1, attentions.shape[0]):
        for j in nb.prange(attentions.shape[1]):
            for k in range(attentions.shape[2]):
                ret[j, k] = np.float32(0.5) * (attentions[i, j, k] + _I) @ ret[j, k]
    return ret


def token_importance(
    attentions: np.ndarray,
    *,
    values: np.ndarray = None,
    window: int = None,
    aggregation: typing.Literal[
        "sum", "mean", "max", "median", "mean_log", "median_log", "Q70", "Q90", "Q95"
    ] = "mean",
    last_layer_only: bool = False,
    rollout: bool = False,
) -> np.ndarray:
    """
    take `attentions` (n_layers × batch_size × num_heads × sequence_length
    × sequence_length) and `values` (n_layers × batch_size × num_heads
    × sequence_length × d_vals) and return token importances (batch_size),
    calculated with a lookahead window of `window`, in a way that is either
    `value_aware` or not;
    `last_layer_only` restricts to the last layer; `rollout` performs attention
    rollout; do not turn both of these on at once
    """
    if last_layer_only:
        a = attentions[-1][np.newaxis]
    elif rollout:
        a = attention_rollout_numba(attentions.astype(np.float32))[np.newaxis]
    else:
        a = attentions
    v = values[-1][np.newaxis] if last_layer_only and values is not None else values
    return agg_str2fn(aggregation)(
        np.sum(
            a
            * (lookahead(a.shape[-1], window) if window is not None else 1)
            * (
                np.linalg.norm(v, axis=-1, ord=1, keepdims=True)
                if values is not None
                else 1
            ),
            axis=3,
        ),
        axis=(0, 2),
    )


def count_top_q(values: list, q: float) -> typing.List[int]:
    """
    takes a ragged list of `values` and returns the number of values exceeding
    the `q`th quantile in each sublist
    """
    values_flat = [v for val in values for v in val]
    qv = np.nanquantile(values_flat, q=q)
    return [int(sum(v >= qv for v in val)) for val in values]


def set_pd_options() -> None:
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.width = 250
    pd.options.display.max_colwidth = 100


if __name__ == "__main__":
    import time

    print(ragged_lists_to_array([[2.0, 3.0], [3.0]]))

    tks = [np.arange(10)]
    tms = [np.array([0] * 3 + [1] * 3 + [2] * 3 + [3])]
    inf = np.array([0] * 3 + [3, 0, 0] + [2] * 3 + [1]).reshape(1, -1)
    print(redact_tokens_times(tks, tms, inf, k=1))
    print(redact_tokens_times(tks, tms, inf, k=1, aggregation="perplexity"))
    print(redact_tokens_times(tks, tms, inf, k=1, method="random"))

    tms_unq, idx = np.unique(tms, return_inverse=True)
    result = np.zeros(shape=tms_unq.shape)
    np.add.at(result, idx, inf.ravel())
    print(result[idx])
    # [0. 0. 0. 3. 3. 3. 6. 6. 6. 1.]

    print(count_top_q([[2, 2, 2, 9], [0, 1], [], [3, 9, 9]], q=0.8))
    # [1, 0, 0, 2]

    rng = np.random.default_rng(42)
    att_eg = np.tril(rng.normal(size=(1000, 8, 16, 4, 32, 32)))

    t0 = time.time()
    for i in range(1000):
        rll = attention_rollout(att_eg[i])
    t1 = time.time()
    print("rollout: {:.2f}".format((t1 - t0)))

    t0 = time.time()
    for i in range(1000):
        rll_nb = attention_rollout_numba(att_eg[i].astype(np.float32))
    t1 = time.time()
    print("rollout numba: {:.2f}".format((t1 - t0)))

    assert np.allclose(rll, rll_nb, rtol=1e-3, atol=1e-1)

    t6 = time.time()
    for i in range(1000):
        tk_imp = token_importance(att_eg[i])
    t7 = time.time()
    print("h2o: {:.2f}".format((t7 - t6)))

    t10 = time.time()
    for i in range(1000):
        tk_imp_sh = token_importance(att_eg[i], window=10)
    t11 = time.time()
    print("sh: {:.2f}".format((t11 - t10)))
