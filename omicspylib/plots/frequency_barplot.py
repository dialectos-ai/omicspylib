"""
Plot the number of missing values per experiment.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from omicspylib import ProteinsDataset
from omicspylib.datasets.abc import TabularDataset


def _plot_barplot(
    x_values: list[str],
    y_values: list[float],
    group: list[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    ax: Axes | None = None,
) -> Axes:
    """
    Draw a colored barplot with annotations.

    Parameters
    ----------
    x_values : list[str]
        List of x values.
    y_values : list[float]
        List of y values.
    group: list[str]
        List of group values.
    title: str
        Title of the plot.
    xlabel: str
        Label for the x-axis.
    ylabel: str
        Label for the y-axis.
    ax: Axes | None
        Matplotlib Axes object to plot on. If None, a new figure and axes are created.

    Returns
    -------
    Axes:
        A pyplot Axes object.

    """
    if not (len(x_values) == len(y_values) == len(group)):
        raise ValueError(
            "x_values, y_values and group labels should be of the same size"
        )

    # setup colors
    unique_categories = list(set(group))
    colors = plt.get_cmap("tab20").colors  # type: ignore
    cmap = {cat: colors[i % len(colors)] for i, cat in enumerate(unique_categories)}
    exp_color = [cmap.get(grp) for grp in group]

    # base plot
    if ax is None:
        _, ax = plt.subplots()

    bar_container = ax.bar(x=x_values, height=y_values, color=exp_color)

    # styles
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.bar_label(bar_container)
    plt.tight_layout()

    return ax


# pylint: disable=too-many-arguments
def plot_missing_values(
    dataset: TabularDataset,
    xlabel: str = "Experiment",
    ylabel: str = "Number of missing values",
    title: str = "Missing values over experiments",
    min_threshold: float = 0,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot the number of missing values per experiment of the dataset.

    Creates a bar plot showing the number of missing values per experiment in the dataset
    with frequency annotation. Returns a matplotlib axes object containing the plot.

    Parameters
    ----------
    dataset: TabularDataset
        Dataset under discussion.
    xlabel: str, optional
        X-axis label.
    ylabel: str, optional
        Y-axis label.
    title: str, optional
        Title of the plot.
    min_threshold: float, optional
        Values below that threshold will be considered as missing values.
    ax: plt.Axes, optional
        If an existing axes object is provided, the plot will be drawn on it.

    Returns
    -------
    ax: plt.Axes
        A matplotlib axes object containing the plot.
    """
    df, n_missing, n_total = dataset.record_counts(
        na_threshold=min_threshold,
        value_type="missing"
    )
    prc_missing = n_missing / n_total * 100

    return _plot_barplot(
        x_values=df["experiment"].tolist(),
        y_values=df["n_counts"].tolist(),
        group=df["condition"].tolist(),
        title=title + f" (~{round(prc_missing, 1)} missing)%",
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
    )


def plot_record_frequency(
    dataset: ProteinsDataset,
    xlabel: str = "Experiment",
    ylabel: str = "Number of records",
    title: str = "Missing records over experiments",
    min_threshold: float = 0,
    ax: Axes | None = None,
) -> Axes:
    df, n_pressent, n_total = dataset.record_counts(
        na_threshold=min_threshold,
        value_type="present"
    )
    prc_missing = n_pressent / n_total * 100

    return _plot_barplot(
        x_values=df["experiment"].tolist(),
        y_values=df["n_counts"].tolist(),
        group=df["condition"].tolist(),
        title=title + f" (~{round(prc_missing, 1)} complete)%",
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
    )
