"""
Plot dataset values as in a density plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
from matplotlib.axes import Axes

from omicspylib import ProteinsDataset


# pylint: disable=too-many-arguments
def plot_density(
        dataset: ProteinsDataset,
        log_transform: bool = False,
        xlabel: str = 'Quantitative value',
        ylabel: str = 'Density',
        title: str = 'Distribution of values across experiments',
        hide_legend: bool = False,
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
    long_data = long_data.loc[long_data['value'] > 0].copy()
    if log_transform:
        long_data['value'] = long_data['value'].apply(np.log2)
    if ax is None:
        _, ax = plt.subplots()

    sns.kdeplot(long_data, x='value', hue='variable', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if hide_legend:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    return ax
