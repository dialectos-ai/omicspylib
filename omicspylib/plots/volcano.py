from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a volcano plot, by plotting on the x-axis the fold change
    (in log2 scale) and on the y-axis the p-value (-log10 transformed).

    A pandas data frame needs to be provided to the function, along with the
    basic values for p-value and fold change. Transformed values will be
    calculated internally.

    Parameters
    ----------
    data: pd.DataFrame
        A pandas data frame containing the data to be plotted.
    pval_col: str
        Column name containing the p-values to be plotted. They
        will be transformed internally with -log10.
    fc_col: str
        Column name containing the fold change of condition a
        over b. It will be transformed internally with log2.
    condition_a: str
        Name of condition a, to be included in the legend and title.
    condition_b: str
        Name of condition b, to be included in the legend and title.
    pval_threshold: float
        Threshold value for considering significant difference.
        By default, values below or equal to 0.05 are considered
        significant.
    fold_change_threshold: float
        Threshold value for considering significant difference in
        fold change. By default, 2x fold difference (from any side)
        is considered significant.
    xmax: float or None
        If ``xmax`` is provided, it will be used to set x limits in the
        range of (-xmax, xmax). Otherwise, it will be calculated base
        on the provided data. It corresponds to transformed fold change
        value.
    ymax: float or None
        If ``ymax`` is provided, it will be used to set the y limit.
        Otherwise, it will be calculated base on the provided data.
        Note that it corresponds to transformed p-values.
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

    minus_log10_pval_col = 'mlog10-pval'
    data[minus_log10_pval_col] = -np.log10(data[pval_col])

    log2_fc_col = 'log2_fc'
    data[log2_fc_col] = np.log2(data[fc_col])

    fc_upper_limit = np.log2(fold_change_threshold)
    fc_lower_limit = -fc_upper_limit

    is_sign_a = (data[pval_col] <= pval_threshold) & (data[log2_fc_col] >= fc_upper_limit)
    is_sign_b = (data[pval_col] <= pval_threshold) & (data[log2_fc_col] <= fc_lower_limit)
    is_non_sign = ~is_sign_a & ~is_sign_b

    sign_a = data.loc[is_sign_a].copy()
    sign_b = data.loc[is_sign_b].copy()
    ns_df = data.loc[is_non_sign].copy()

    if xmax is None:
        x_expansion_factor = 1.1
        xmax = data[log2_fc_col].max() * x_expansion_factor
    if ymax is None:
        ymax = data[minus_log10_pval_col].max()

    if ax is None:
        _, ax = plt.subplots()

    ax.vlines(x=-1, ymin=0, ymax=ymax, color='grey', linestyles=':', alpha=0.3)
    ax.vlines(x=1, ymin=0, ymax=ymax, color='grey', linestyles=':', alpha=0.3)
    ax.hlines(y=-np.log10(0.05), xmin=-xmax, xmax=xmax, linestyles=':', alpha=0.3, color='grey')
    ax.scatter(ns_df[log2_fc_col], ns_df[minus_log10_pval_col], label='non-significant', color='grey')
    ax.scatter(sign_a[log2_fc_col], sign_a[minus_log10_pval_col], label=condition_a, color=color_a)
    ax.scatter(sign_b[log2_fc_col], sign_b[minus_log10_pval_col], label=condition_b, color=color_b)

    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel(f'Fold change {condition_a}/{condition_b} (log2 scale)')
    ax.set_ylabel('t-test P-value (-log10)')
    ax.annotate('p<0.05', xy=(-xmax * 0.99, -np.log10(0.05)))
    ax.set_title(f'Significantly differently abundant proteins \n '
                 f'between {condition_a} and {condition_b}')
    ax.legend()

    return ax
