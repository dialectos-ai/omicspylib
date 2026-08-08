"""
Plot dataset values as in a density plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from omicspylib import ProteinsDataset


# pylint: disable=too-many-arguments
def plot_density(
        dataset: ProteinsDataset,
        log_transform: bool = False,
        xlabel: str = 'Quantitative value',
        ylabel: str = 'Density',
        title: str = 'Distribution of values across experiments',
        hide_legend: bool = False,
        color_by_group: bool = True,
        ax: Axes | None = None) -> Axes:
    """
    Generic function for creating a density plot over quantitative
    values of a dataset. It returns a matplotlib axes object that you can
    further customize. For more detailed customization, call the ``.to_table()``
    method on the dataset object and create a plot based on your needs.

    By default, 0s and nan values are removed.

    Parameters
    ----------
    dataset: ProteinsDataset
        A proteins dataset object.
    log_transform: bool
        If specified, values will be transformed to log2.
    xlabel: str
        X axis label.
    ylabel: str
        Y axis label.
    title: str
        Plot title.
    hide_legend: bool
        If set to ``True``, the legend will be removed.
    color_by_group: bool
        If True, experiments belonging to the same condition group will share the same color.
    ax: Axes | None
        You can provide a plt.Axes object to create a plot
        on that. Otherwise, a new object will be created and returned.

    Returns
    -------
    Axes
        A matplotlib Axes object.
    """
    tabular_dataset = dataset.to_table()
    long_data = tabular_dataset.melt()
    long_data = long_data.loc[long_data["value"] > 0].copy()
    if log_transform:
        long_data["value"] = long_data["value"].apply(np.log2)

    if ax is None:
        _, ax = plt.subplots()

    if color_by_group:
        counts_df, _, _ = dataset.record_counts()
        exp_to_cond = dict(zip(counts_df["experiment"], counts_df["condition"], strict=True))
        long_data["condition"] = long_data["variable"].map(exp_to_cond)

        unique_conditions = list(set(long_data["condition"].dropna()))
        colors = plt.get_cmap("tab20").colors  # type: ignore
        cmap = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_conditions)}
        exp_palette = {exp: cmap[cond] for exp, cond in exp_to_cond.items() if cond in cmap}

        sns.kdeplot(long_data, x="value", hue="variable", palette=exp_palette, common_norm=False, ax=ax)

        if hide_legend:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        else:
            # Rebuild legend showing one entry per group/condition
            handles = [Line2D([0], [0], color=cmap[cond], lw=2, label=cond) for cond in unique_conditions]
            ax.legend(handles=handles, title="condition")
    else:
        sns.kdeplot(long_data, x="value", hue="variable", legend=not hide_legend, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax
