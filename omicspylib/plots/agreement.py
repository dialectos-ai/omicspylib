"""
Plots of the agreement between experiments: a correlation heatmap of all
pairs, and a scatter or Bland-Altman plot of one pair.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from omicspylib.calculations.bland_altman import calc_bland_altman
from omicspylib.calculations.correlation import CorrelationMethod, calc_pairwise_correlation
from omicspylib.datasets.abc import MissingHandling, TabularDataset
from omicspylib.docs.definition import doc

PairPlotType = Literal["scatter", "bland_altman"]

_MAX_EXPERIMENTS_WITH_VALUES = 12
_TEXT_BOX = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"}


# pylint: disable=too-many-arguments
@doc()
def plot_correlation_heatmap(
    dataset: TabularDataset,
    method: CorrelationMethod = "spearman",
    missing: MissingHandling = "pairwise",
    log_transform: bool | None = None,
    min_records: int = 10,
    na_threshold: float = 0.0,
    show_values: bool | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Heatmap of the correlation coefficient between every pair of experiments.

    Experiments are ordered by condition, and lines separate the conditions,
    so replicates of the same condition form blocks along the diagonal.
    These blocks should be brighter (higher correlation) than the rest of
    the heatmap. An experiment that correlates better with another
    condition than with its own replicates may be swapped or failed.

    The coefficients are calculated with
    :meth:`~omicspylib.datasets.abc.TabularDataset.correlation`, using
    ``pairs="all"``.

    Parameters
    ----------
    {dataset}
    method : {{"spearman", "pearson", "kendall"}}, optional
        Correlation coefficient. Default is ``"spearman"``.
    missing : {{"pairwise", "listwise"}}, optional
        Which records to use. See ``correlation``.
    log_transform : bool or None, optional
        Log2 transform the values first. ``None`` (default) means ``True``
        for ``"pearson"`` only. See ``correlation``.
    min_records : int, optional
        Pairs with fewer records are left empty. Default is ``10``.
    {na_threshold}
    show_values : bool or None, optional
        Write the coefficient in each cell. ``None`` (default) writes them
        only if there are up to 12 experiments, where they remain readable.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"viridis"``.
    title : str, optional
        Plot title. By default, it names the correlation method.
    {ax}
    **kwargs
        Passed to :func:`seaborn.heatmap`.

    {returns_ax}

    See Also
    --------
    plot_experiment_pair : Scatter or Bland-Altman plot of one pair.
    """
    corr = dataset.correlation(
        method=method,
        pairs="all",
        missing=missing,
        log_transform=log_transform,
        min_records=min_records,
        na_threshold=na_threshold,
    )
    experiments = dataset.experiment_names()
    matrix = pd.DataFrame(np.nan, index=experiments, columns=experiments)
    for exp_a, exp_b, coef in zip(
        corr["experiment_a"], corr["experiment_b"], corr["coefficient"], strict=True
    ):
        matrix.at[exp_a, exp_b] = coef
        matrix.at[exp_b, exp_a] = coef
    for exp in experiments:
        matrix.at[exp, exp] = 1.0

    if ax is None:
        _, ax = plt.subplots()

    if show_values is None:
        show_values = len(experiments) <= _MAX_EXPERIMENTS_WITH_VALUES
    heatmap_kwargs = {"annot": show_values, "fmt": ".2f", "cmap": cmap, "square": True}
    heatmap_kwargs.update(kwargs)
    sns.heatmap(matrix, ax=ax, **heatmap_kwargs)

    # separate conditions
    boundary = 0
    for cond in dataset.condition_names[:-1]:
        boundary += len(dataset.experiment_names(cond))
        ax.axhline(boundary, color="white", linewidth=2)
        ax.axvline(boundary, color="white", linewidth=2)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title if title is not None else f"{method.capitalize()} correlation")
    return ax


# pylint: disable=too-many-arguments,too-many-locals
@doc()
def plot_experiment_pair(
    dataset: TabularDataset,
    experiment_a: str,
    experiment_b: str,
    plot_type: PairPlotType = "scatter",
    log_transform: bool = True,
    na_threshold: float = 0.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Compare the values of two experiments, record by record.

    Only the records with a value in both experiments are plotted. Missing
    values (stored as ``0`` or ``NaN``) are never plotted as measurements.

    * ``plot_type="scatter"``: ``experiment_b`` against ``experiment_a``,
      with the identity line. Points close to the line mean good agreement.
      The Spearman correlation is shown in the plot.
    * ``plot_type="bland_altman"``: the difference ``a - b`` against the
      mean of the two, for each record. A solid line shows the bias (mean
      difference) and dashed lines the limits of agreement (bias ± 1.96
      standard deviations). On log2 values the difference is a log2 fold
      change, so a bias away from ``0`` is a systematic offset between the
      experiments, and a wider spread at low intensities shows that
      low-abundance records are less reproducible.

    To compare several pairs, call it once per pair, e.g., on a grid of
    axes created with :func:`matplotlib.pyplot.subplots`.

    Parameters
    ----------
    {dataset}
    experiment_a : str
        Name of the first experiment.
    experiment_b : str
        Name of the second experiment.
    plot_type : {{"scatter", "bland_altman"}}, optional
        Type of plot. Default is ``"scatter"``.
    log_transform : bool, optional
        Log2 transform the values first. Default is ``True``. Set to
        ``False`` if the dataset is already log-transformed.
    {na_threshold}
    xlabel : str, optional
        X axis label. By default, it is derived from the plot type.
    ylabel : str, optional
        Y axis label. By default, it is derived from the plot type.
    title : str, optional
        Plot title. By default, it names the two experiments.
    {ax}
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.scatter`, e.g., ``s``,
        ``alpha`` or ``color``.

    {returns_ax}

    See Also
    --------
    plot_correlation_heatmap : Correlation of all pairs of experiments.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> pairs = dataset.correlation()  # doctest: +SKIP
    >>> fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # doctest: +SKIP
    >>> for ax, (_, pair) in zip(axes, pairs.head(3).iterrows()):  # doctest: +SKIP
    ...     plot_experiment_pair(
    ...         dataset, pair["experiment_a"], pair["experiment_b"],
    ...         plot_type="bland_altman", ax=ax,
    ...     )
    """
    if plot_type not in ("scatter", "bland_altman"):
        raise ValueError(
            f"plot_type must be 'scatter' or 'bland_altman', got {plot_type!r}."
        )

    # pylint: disable=protected-access
    table, _ = dataset._paired_values(
        [(experiment_a, experiment_b)], "pairwise", log_transform, na_threshold
    )
    values = table[[experiment_a, experiment_b]].dropna()
    a = values[experiment_a]
    b = values[experiment_b]
    unit = " (log2)" if log_transform else ""

    if ax is None:
        _, ax = plt.subplots()

    scatter_kwargs = {"s": 8, "alpha": 0.5, "linewidths": 0}
    scatter_kwargs.update(kwargs)

    if plot_type == "scatter":
        ax.scatter(a, b, **scatter_kwargs)
        if len(values) > 0:
            low = min(a.min(), b.min())
            high = max(a.max(), b.max())
            ax.plot([low, high], [low, high], color="grey", linestyle="--", linewidth=1)
        corr = calc_pairwise_correlation(
            values, [(experiment_a, experiment_b)], method="spearman", min_records=2
        ) if len(values) >= 2 else None
        if corr is not None:
            ax.text(
                0.03, 0.97,
                f"Spearman = {corr['coefficient'].iloc[0]:.3f}\nn = {len(values)}",
                transform=ax.transAxes, va="top", ha="left", bbox=_TEXT_BOX,
            )
        default_xlabel = f"{experiment_a}{unit}"
        default_ylabel = f"{experiment_b}{unit}"
    else:
        ax.scatter((a + b) / 2, a - b, **scatter_kwargs)
        stats = calc_bland_altman(values, [(experiment_a, experiment_b)], min_records=2)
        row = stats.iloc[0]
        if not np.isnan(row["bias"]):
            ax.axhline(row["bias"], color="black", linewidth=1)
            ax.axhline(row["loa_lower"], color="grey", linestyle="--", linewidth=1)
            ax.axhline(row["loa_upper"], color="grey", linestyle="--", linewidth=1)
            ax.text(
                0.03, 0.97,
                f"bias = {row['bias']:.3f}\n"
                f"limits = [{row['loa_lower']:.3f}, {row['loa_upper']:.3f}]\n"
                f"n = {int(row['n_records'])}",
                transform=ax.transAxes, va="top", ha="left", bbox=_TEXT_BOX,
            )
        default_xlabel = f"Mean{unit}"
        default_ylabel = f"Difference, {experiment_a} - {experiment_b}{unit}"

    ax.set_xlabel(xlabel if xlabel is not None else default_xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)
    ax.set_title(title if title is not None else f"{experiment_a} vs {experiment_b}")
    return ax
