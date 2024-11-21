"""
Venn diagram plots.
"""
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn2  # type: ignore


def plot_venn2(data: pd.DataFrame,
               condition_a: str,
               condition_b: str,
               color_a: str = 'blue',
               color_b: str = 'red',
               title: str = 'Venn Diagram',
               ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Venn diagram between two groups.

    Parameters
    ----------
    data
    condition_a
    condition_b
    color_a
    color_b
    title
    ax

    Returns
    -------
    plt.Axes
        Matplotlib's Axes object.
    """
    # extract frequencies
    f_counts = data[['frequency_class']]\
        .reset_index()\
        .groupby('frequency_class')\
        .count()
    counts_col = f_counts.columns.tolist()[-1]
    grp1_idx = [i for i in f_counts.index if i.endswith(condition_a)]
    if len(grp1_idx) > 0:
        f_a = f_counts.loc[grp1_idx[0], counts_col]
    else:
        f_a = 0

    grp2_idx = [i for i in f_counts.index if i.endswith(condition_b)]
    if len(grp2_idx) > 0:
        f_b = f_counts.loc[grp2_idx[0], counts_col]
    else:
        f_b = 0

    f_common = f_counts.loc['common', counts_col]

    # plot venn
    if ax is None:
        _, ax = plt.subplots()
    venn2(subsets=(f_a, f_b, f_common),
          set_labels=(condition_a, condition_b),
          ax=ax,
          set_colors=(color_a, color_b))

    # stylize
    ax.set_title(title)
    return ax
