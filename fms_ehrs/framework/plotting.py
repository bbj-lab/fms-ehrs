#!/usr/bin/env python3

"""
functions for plotting
"""

import collections
import pathlib

import numpy as np
from plotly import express as px
from plotly import graph_objects as go
from plotly import io as pio
from sklearn import calibration as skl_cal
from sklearn import metrics as skl_mets

from fms_ehrs.framework.util import Dictlike, Pathlike

pio.kaleido.scope.mathjax = None

mains = ("#EAAA00", "#DE7C00", "#789D4A", "#275D38", "#007396", "#56315F", "#A4343A")
lights = ("#F3D03E", "#ECA154", "#A9C47F", "#9CAF88", "#3EB1C8", "#86647A", "#B46A55")
darks = ("#CC8A00", "#A9431E", "#13301C", "#284734", "#002A3A", "#41273B", "#643335")
colors = mains + lights + darks


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
                name="{} (Brier: {:.3f})".format(
                    name, skl_mets.brier_score_loss(y_true=y_true, y_prob=y_score)
                ),
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
        font_family="CMU Serif, Times New Roman, serif",
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
                name="{} (AUC: {:.3f})".format(
                    name, skl_mets.roc_auc_score(y_true=y_true, y_score=y_score)
                ),
                marker=dict(color=colors[(i + 1) % len(colors)], size=1),
            )
        )

    fig.update_layout(
        title="Receiver operating characteristic",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        font_family="CMU Serif, Times New Roman, serif",
    )

    if savepath is None:
        fig.show()
    else:
        fig.write_image(pathlib.Path(savepath).expanduser().resolve())


def plot_precision_recall_curve(
    named_results: Dictlike, savepath: Pathlike = None, decimals: int = 3
):
    """
    plot a precision-recall curve for each named set of predictions;
    {"name": {"y_true": y_true, "y_score": y_score}}
    if provided a `savepath`; otherwise, display
    """

    fig = go.Figure()

    for i, (name, results) in enumerate(named_results.items()):

        y_true = results["y_true"]
        y_score = results["y_score"]

        assert y_true.shape[0] == y_score.shape[0]

        precs, recs, _ = skl_mets.precision_recall_curve(
            y_true, np.round(y_score, decimals=decimals), drop_intermediate=True
        )

        fig.add_trace(
            go.Scatter(
                x=recs,
                y=precs,
                mode="lines+markers",
                name="{} (PR-AUC: {:.3f})".format(name, skl_mets.auc(recs, precs)),
                marker=dict(color=colors[(i + 1) % len(colors)], size=1),
            )
        )

    fig.update_layout(
        title="Precision-recall curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        font_family="CMU Serif, Times New Roman, serif",
    )

    if savepath is None:
        fig.show()
    else:
        fig.write_image(pathlib.Path(savepath).expanduser().resolve())


def plot_histogram(
    arr: np.array, title: str = "Histogram", nbins: int = 50, savepath: Pathlike = None
):
    """
    plot a histogram of the non-nan values in an array `arr`;
    if provided a `savepath`; otherwise, display
    """

    fig = px.histogram(
        arr[np.isfinite(arr)].ravel(), nbins=nbins, labels={"value": "Value"}
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=False,
        font_family="CMU Serif, Times New Roman, serif",
    )

    if savepath is None:
        fig.show()
    else:
        fig.write_image(pathlib.Path(savepath).expanduser().resolve())


def plot_histograms(
    named_arrs: dict,
    *,
    title: str = "Histogram",
    nbins: int = 50,
    savepath: Pathlike = None,
    **kwargs,
):
    """
    plot a histogram of the non-nan values in an array `arr`;
    if provided a `savepath`; otherwise, display;
    NB: by default, plotly saves all data passed to its histogram function--not simply
    the summary statistics required to create the plot
    """

    fig = go.Figure()

    edges = np.histogram_bin_edges(
        np.concatenate([x[np.isfinite(x)].ravel() for x in named_arrs.values()]),
        bins=nbins,
    )

    for i, (name, arr) in enumerate(named_arrs.items()):
        ct, bins = np.histogram(arr[np.isfinite(arr)].ravel(), bins=edges, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_widths = bins[1:] - bins[:-1]
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=ct,
                name=name,
                opacity=0.5,
                width=bin_widths,
                marker_color=colors[(i + 1) % len(colors)],
            )
        )

    fig.update_layout(
        barmode="overlay",
        template="plotly_white",
        title=title,
        font_family="CMU Serif, Times New Roman, serif",
        **kwargs,
    )

    if savepath is None:
        fig.show()
    else:
        fig.write_image(pathlib.Path(savepath).expanduser().resolve())


def imshow_text(
    values: np.array,
    text: np.array,
    title: str = "",
    savepath=None,
    autocolor_text: bool = False,
    **layout_kwargs,
):
    assert values.shape == text.shape
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12} | ({} if autocolor_text else {"color": "black"}),
            colorscale=px.colors.sequential.Viridis[4:],
            reversescale=False,
            showscale=True,
            zsmooth=False,
            xgap=1,
            ygap=1,
        )
    )

    fig.update_layout(
        # autosize=False,
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"
        ),
        # height=3000,
        # width=1000,
        font_family="CMU Serif, Times New Roman, serif",
        **layout_kwargs,
    )

    if savepath is None:
        fig.show()
    else:
        fig.write_image(pathlib.Path(savepath).expanduser().resolve())


if __name__ == "__main__":

    import pandas as pd

    from fms_ehrs.framework.stats import generate_classifier_preds

    rng: np.random._generator.Generator = np.random.default_rng(seed=42)
    num_preds = 3
    y_true, y_preds = generate_classifier_preds(num_preds=num_preds)

    named_results = collections.OrderedDict()
    for i in range(num_preds):
        named_results[f"test{i}"] = {"y_true": y_true, "y_score": y_preds[i]}

    plot_calibration_curve(named_results)
    plot_roc_curve(named_results)
    plot_precision_recall_curve(named_results)

    vals = rng.normal(scale=0.2, size=10_000).reshape((10, 10, -1))
    vals[vals > 0.6] = np.nan
    plot_histogram(vals)

    plot_histograms({"foo": vals, "bar": vals + 0.2}, xaxis_title="bits")

    n_tot, n_col = 102, 6
    vals = rng.poisson(lam=20, size=n_tot).reshape((-1, n_col))
    text = np.arange(n_tot).astype(str).reshape((-1, n_col))
    imshow_text(
        values=vals,
        text=text,
        width=1000,
        height=600,
        autosize=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    results = pd.DataFrame(
        {
            "pct": [0, 0.5, 1, 2.5, 5],
            "same admission mortality": [0.914, 0.920, 0.922, 0.911, 0.892],
            "long length of stay": [0.750, 0.766, 0.776, 0.751, 0.727],
            "ICU admission": [0.615, 0.757, 0.805, 0.796, 0.766],
            "IMV event": [0.675, 0.805, 0.796, 0.835, 0.808],
        }
    ).melt(id_vars="pct", var_name="outcome", value_name="AUC")

    fig = px.line(
        results,
        x="pct",
        y="AUC",
        color="outcome",
        title="Overall AUC (test set) vs. Fraction of local data used for finetuning",
        color_discrete_sequence=colors[1:],
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Percentage of local data used for finetuning",
        yaxis_title="overall AUC (local test set)",
        template="plotly_white",
        font_family="CMU Serif, Times New Roman, serif",
    )
    fig.show()
    # fig.write_image(pathlib.Path("~/Downloads/sft_perf.pdf").expanduser().resolve())
