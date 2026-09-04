"""
Venn diagram plots.
"""
from typing import cast

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib_venn import venn2  # type: ignore

from omicspylib.docs.definition import doc


@doc()
def plot_venn2(data: pd.DataFrame,
               condition_a: str,
               condition_b: str,
               color_a: str = "blue",
               color_b: str = "red",
               title: str = "Venn Diagram",
               ax: Axes | None = None) -> Axes:
    """
    Venn diagram between two groups.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing a 'frequency_class' column that categorizes records
        as belonging to condition_a, condition_b, or both ('common').
    condition_a : str
        Name/label of the first condition.
    condition_b : str
        Name/label of the second condition.
    color_a : str, default='blue'
        Color of the circle corresponding to condition_a.
    color_b : str, default='red'
        Color of the circle corresponding to condition_b.
    {title}
    {ax}

    {returns_ax}
    """
    # extract frequencies
    f_counts = data[["frequency_class"]]\
        .reset_index()\
        .groupby("frequency_class")\
        .count()
    counts_col = f_counts.columns.tolist()[-1]
    grp1_idx = [i for i in f_counts.index if i.endswith(condition_a)]
    if len(grp1_idx) > 0:
        f_a = cast(float, f_counts.at[grp1_idx[0], counts_col])
    else:
        f_a = 0

    grp2_idx = [i for i in f_counts.index if i.endswith(condition_b)]
    if len(grp2_idx) > 0:
        f_b = cast(float, f_counts.at[grp2_idx[0], counts_col])
    else:
        f_b = 0

    f_common = cast(float, f_counts.at["common", counts_col])

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
