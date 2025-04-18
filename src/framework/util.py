#!/usr/bin/env python3

"""
utility functions
"""

import collections
import logging
import os
import pathlib
import typing

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import sklearn.metrics as skl_mets
import sklearn.calibration as skl_cal
import torch as t

Pathlike: typing.TypeAlias = pathlib.PurePath | str | os.PathLike
Dictlike: typing.TypeAlias = collections.OrderedDict | dict

pio.kaleido.scope.mathjax = None

mains = ("#EAAA00", "#DE7C00", "#789D4A", "#275D38", "#007396", "#56315F")
lights = ("#F3D03E", "#ECA154", "#A9C47F", "#9CAF88", "#3EB1C8", "#86647A")
darks = ("#CC8A00", "#A9431E", "#13301C", "#284734", "#002A3A", "#41273B")
colors = mains + lights + darks


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


def plot_calibration_curve(
    named_results: Dictlike, n_bins: int = 10, savepath: Pathlike = None
):
    """
    plot a calibration curve for each named set of predictions;
    {"name": {"y_true": y_true, "y_score": y_score}}
    if provided a `savepath`; otherwise, display
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash", color="gray"),
        )
    )

    for i, (name, results) in enumerate(named_results.items()):

        y_true = results["y_true"]
        y_score = results["y_score"]

        assert y_true.shape[0] == y_score.shape[0]

        prob_true, prob_pred = skl_cal.calibration_curve(y_true, y_score, n_bins=n_bins)

        fig.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="{} Calibration".format(name),
                marker=dict(color=colors[i % len(colors)]),
            )
        )

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )

    if savepath is None:
        fig.show()
    else:
        fig.write_image(pathlib.Path(savepath).expanduser().resolve())


def plot_roc_curve(named_results: Dictlike, savepath: Pathlike = None):
    """
    plot a ROC curve for each named set of predictions;
    {"name": {"y_true": y_true, "y_score": y_score}}
    if provided a `savepath`; otherwise, display
    """

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(dash="dash", color="gray"),
        )
    )

    for i, (name, results) in enumerate(named_results.items()):

        y_true = results["y_true"]
        y_score = results["y_score"]

        assert y_true.shape[0] == y_score.shape[0]

        fpr, tpr, _ = skl_mets.roc_curve(y_true, y_score)

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines+markers",
                name="ROC (AUC: {:.3f})".format(
                    skl_mets.roc_auc_score(y_true=y_true, y_score=y_score)
                ),
                marker=dict(color=colors[i % len(colors)], size=3),
            )
        )

    fig.update_layout(
        title="Receiver operating characteristic",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )

    if savepath is None:
        fig.show()
    else:
        fig.write_image(pathlib.Path(savepath).expanduser().resolve())


def ragged_lists_to_array(ls_arr: list[np.array]) -> np.array:
    """
    form an 2d-array from a collection of variably-sized 1d-arrays
    """
    n, m = len(ls_arr), max(map(len, ls_arr))
    arr = np.full(shape=(n, m), fill_value=np.nan)
    for i, x in enumerate(ls_arr):
        arr[i, : len(x)] = x
    return arr


def set_pd_options():
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.width = 250
    pd.options.display.max_colwidth = 100


if __name__ == "__main__":
    from src.framework.logger import get_logger

    logger = get_logger()
    np_rng = np.random.default_rng(42)

    print(ragged_lists_to_array([[2.0, 3.0], [3.0]]))

    y_seed = np_rng.uniform(size=1000)
    y_true = (y_seed > 0.4).astype(int)
    y_pred = np.clip(y_seed + np_rng.normal(scale=0.2, size=1000), a_min=0, a_max=1)
    y_pred2 = np.clip(y_seed + np_rng.normal(scale=0.2, size=1000), a_min=0, a_max=1)
    log_classification_metrics(y_true, y_pred, logger)

    named_results = collections.OrderedDict()
    named_results["test1"] = {"y_true": y_true, "y_score": y_pred}
    named_results["test2"] = {"y_true": y_true, "y_score": y_pred2}

    plot_calibration_curve(named_results)
    plot_roc_curve(named_results)
