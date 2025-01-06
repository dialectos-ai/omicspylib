"""
Volcano plot for the comparison of two groups.
"""
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

_MINUS_LOG10_PVAL_COL = 'mlog10-pval'
_LOG2_FC_COL = 'log2_fc'


# pylint: disable=too-many-arguments,too-many-locals
def plot_volcano(
        data: pd.DataFrame,
        pval_col: str,
        fc_col: str,
        condition_a: str,
        condition_b: str,
        color_a: str = 'blue',
        color_b: str = 'red',
        pval_threshold: float = 0.05,
        fold_change_threshold: float = 2.0,
        xmax: Optional[float] = None,
        ymax: Optional[float] = None,
        xlabel: Optional[str] = None,
        ylabel: str = 't-test P-value (-log10)',
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a volcano plot, by plotting on the x-axis the fold change
    (in a log2 scale) and on the y-axis the p-value (-log10 transformed).

    A Pandas data frame needs to be provided to the function, along with the
    basic values for p-value and fold change. Transformed values will be
    calculated internally.

    Parameters
    ----------
    data: pd.DataFrame
        A Pandas data frame containing the data to be plotted.
    pval_col: str
        Column name containing the p-values to be plotted. They
        will be transformed internally with -log10.
    fc_col: str
        Column name containing the fold change of condition ``a``
        over ``b``. It will be transformed internally with log2.
    condition_a: str
        Name of "condition a" to be included in the legend and title.
    condition_b: str
        Name of "condition b" to be included in the legend and title.
    color_a: str
        Color for "condition a".
    color_b: str
        Color for "condition b".
    pval_threshold: float
        Threshold value for considering significant difference.
        By default, the values below or equal to 0.05 are considered
        significant.
    fold_change_threshold: float
        Threshold value for considering significant difference in
        fold change. By default, 2x fold difference (from any side)
        is considered significant.
    xmax: float or None
        If ``xmax`` is provided, it will be used to set x limits in the
        range of (-xmax, xmax). Otherwise, it will be calculated based
        on the provided data. It corresponds to transformed fold change
        value.
    ymax: float or None
        If ``ymax`` is provided, it will be used to set the y limit.
        Otherwise, it will be calculated based on the provided data.
        Note that it corresponds to transformed p-values.
    xlabel: str
        If provided will replace the default x-label.
    ylabel: str
        If provided will replace the default y-label.
    title: str
        If provided will replace the default title.
    ax: plt.Axes or None
        If a matplotlib axes object is provided, the plot will be drawn
        on it. Otherwise, a new plt.Axes object is created and returned
        to the user.

    Returns
    -------
    plt.Axes
        The plt.Axes object where the plot is drawn.
    """
    data = data.copy()

    ns_df, sign_a, sign_b = _split_dataset_into_subsets(
        data, fc_col, fold_change_threshold, pval_col, pval_threshold)

    x_max, y_max = _define_xy_limits(data, xmax, ymax)

    if ax is None:
        _, ax = plt.subplots()

    _plot_lines_and_data(ax,
                         color_a, color_b,
                         condition_a, condition_b,
                         ns_df, sign_a, sign_b,
                         x_max, y_max)

    _set_plot_annotations(ax, condition_a, condition_b,
                          title, x_max, xlabel, ylabel)

    return ax


def _set_plot_annotations(
        ax, condition_a, condition_b,
        title, x_max, xlabel, ylabel):
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f'Fold change {condition_a}/{condition_b} (log2 scale)')
    ax.set_ylabel(ylabel)
    ax.annotate('p<0.05', xy=(-x_max * 0.99, -np.log10(0.05)))
    if title:
        ax.set_title(f'Significantly differently abundant proteins \n '
                     f'between {condition_a} and {condition_b}')
    else:
        ax.set_title(title)
    ax.legend()


def _plot_lines_and_data(ax, color_a, color_b,
                         condition_a, condition_b,
                         ns_df, sign_a, sign_b,
                         x_max, y_max):
    for x_val in [-1, 1]:
        ax.vlines(x=x_val, ymin=0,
                  ymax=y_max,
                  color='grey',
                  linestyles=':',
                  alpha=0.3)
    ax.hlines(y=-np.log10(0.05),
              xmin=-x_max,
              xmax=x_max,
              linestyles=':',
              alpha=0.3,
              color='grey')
    ax.scatter(ns_df[_LOG2_FC_COL],
               ns_df[_MINUS_LOG10_PVAL_COL],
               label='non-significant',
               color='grey')
    ax.scatter(sign_a[_LOG2_FC_COL],
               sign_a[_MINUS_LOG10_PVAL_COL],
               label=condition_a,
               color=color_a)
    ax.scatter(sign_b[_LOG2_FC_COL],
               sign_b[_MINUS_LOG10_PVAL_COL],
               label=condition_b,
               color=color_b)
    ax.set_xlim(-x_max, x_max)


def _split_dataset_into_subsets(
        data, fc_col, fold_change_threshold, pval_col,
        pval_threshold):
    data[_MINUS_LOG10_PVAL_COL] = -np.log10(data[pval_col])
    data[_LOG2_FC_COL] = np.log2(data[fc_col])
    fc_upper_limit = np.log2(fold_change_threshold)
    fc_lower_limit = -fc_upper_limit
    is_sign_a = (data[pval_col] <= pval_threshold) & (data[_LOG2_FC_COL] >= fc_upper_limit)
    is_sign_b = (data[pval_col] <= pval_threshold) & (data[_LOG2_FC_COL] <= fc_lower_limit)
    is_non_sign = ~is_sign_a & ~is_sign_b
    sign_a = data.loc[is_sign_a].copy()
    sign_b = data.loc[is_sign_b].copy()
    ns_df = data.loc[is_non_sign].copy()
    return ns_df, sign_a, sign_b


def _define_xy_limits(data, xmax, ymax):
    """
    Specify the x, y max limits.
    I assume that you want a symetrical plot, so for x_min use -x_max.
    """
    if xmax is None:
        x_expansion_factor = 1.1
        x_max = data[_LOG2_FC_COL].max() * x_expansion_factor
    else:
        x_max = xmax
    if ymax is None:
        y_max = data[_MINUS_LOG10_PVAL_COL].max()
    else:
        y_max = ymax
    return x_max, y_max
