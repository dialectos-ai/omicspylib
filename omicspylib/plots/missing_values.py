"""
Plot number of missing values per experiment.
"""
from typing import Optional

import matplotlib.pyplot as plt

from omicspylib import ProteinsDataset


def plot_missing_values(
        dataset: ProteinsDataset,
        xlabel: str = 'Experiment',
        ylabel: str = 'Number of missing values',
        title: str = 'Missing values over experiments',
        min_threshold: float = 0,
        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot number of missing values per experiment of the dataset.

    Parameters
    ----------
    dataset: ProteinsDataset
        Dataset under discussion.
    xlabel: str, optional
        X-axis label.
    ylabel: str, optional
        Y-axis label.
    title: str, optional
        Title of the plot.
    min_threshold: float, optional
        Values below that threshold, will be considered as missing values.
    ax: plt.Axes, optional
        If an existing axes object is provided, the plot will be drawn on it.

    Returns
    -------
    ax: plt.Axes
        A matplotlib axes object containing the plot.
    """
    df, n_missing, n_total = dataset.missing_values(min_threshold)
    prc_missing = n_missing / n_total * 100

    if ax is None:
        _, ax = plt.subplots()

    colors = plt.get_cmap('tab20').colors  # type: ignore
    unique_categories = df['condition'].unique()
    cmap = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
    color_col = df['condition'].map(cmap)

    bar_container = ax.bar(df['experiment'], df['n_missing'], color=color_col)
    ax.set_title(title + f' (~{round(prc_missing, 1)} missing)%')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(df['experiment'], rotation=45, ha='right')
    ax.bar_label(bar_container)
    plt.tight_layout()

    return ax
