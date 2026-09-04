"""
Plot the number of missing values per experiment.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from omicspylib import ProteinsDataset
from omicspylib.datasets.abc import TabularDataset
from omicspylib.docs.definition import doc
from omicspylib.plots.utils import apply_text_annotation_formatting, apply_xtick_formatting


def _plot_barplot(
    x_values: list[str],
    y_values: list[float],
    group: list[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    ax: Axes | None = None,
    **kwargs
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
    {kwargs_doc}

    Returns
    -------
    Axes:
        A pyplot Axes object.

    """
    if not (len(x_values) == len(y_values) == len(group)):
        raise ValueError(
            "x_values, y_values and group labels should be of the same size"
        )

    text_annotation_size = kwargs.get("text_annotation_size", None)
    text_annotation_rotation = kwargs.get("text_annotation_rotation", 0)
    text_annotation_round_digits = kwargs.get("text_annotation_round_digits", None)
    text_xlabel_rotation = kwargs.get("text_xlabel_rotation", 45)
    text_xlabel_ha = kwargs.get("text_xlabel_ha", None)
    text_xlabel_va = kwargs.get("text_xlabel_va", None)

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
    ax.set_xticklabels(x_values)

    apply_xtick_formatting(
        ax,
        rotation=text_xlabel_rotation,
        ha=text_xlabel_ha,
        va=text_xlabel_va,
        default_rotation=45,
    )

    bar_label_kwargs = {}
    if text_annotation_size is not None:
        bar_label_kwargs["fontsize"] = text_annotation_size
    if text_annotation_rotation != 0:
        bar_label_kwargs["rotation"] = text_annotation_rotation

    ax.bar_label(bar_container, **bar_label_kwargs)

    apply_text_annotation_formatting(
        ax,
        rotation=text_annotation_rotation,
        round_digits=text_annotation_round_digits,
    )

    plt.tight_layout()

    return ax


# pylint: disable=too-many-arguments
@doc(
    "text_annotation_size",
    "text_annotation_rotation",
    "text_annotation_round_digits",
    "text_xlabel_rotation",
    "text_xlabel_ha",
    "text_xlabel_va",
)
def plot_missing_values(
    dataset: TabularDataset,
    xlabel: str = "Experiment",
    ylabel: str = "Number of missing values",
    title: str = "Missing values over experiments",
    min_threshold: float = 0,
    ax: Axes | None = None,
    **kwargs
) -> Axes:
    """
    Plot the number of missing values per experiment of the dataset.

    Creates a bar plot showing the number of missing values per experiment in the dataset
    with frequency annotation. Returns a matplotlib axes object containing the plot.

    Parameters
    ----------
    {dataset}
    xlabel: str
        X-axis label.
    ylabel: str
        Y-axis label.
    title: str
        Title of the plot.
    {min_threshold}
    {ax}
    {kwargs_doc}

    {returns_ax}
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
        title=title + f" (~{round(prc_missing, 1)}% missing)",
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        **kwargs
    )


@doc(
    "text_annotation_size",
    "text_annotation_rotation",
    "text_xlabel_rotation",
    "text_xlabel_ha",
    "text_xlabel_va",
)
def plot_record_frequency(
    dataset: ProteinsDataset,
    xlabel: str = "Experiment",
    ylabel: str = "Number of records",
    title: str = "Missing records over experiments",
    min_threshold: float = 0,
    ax: Axes | None = None,
    **kwargs
) -> Axes:
    """Plot record counts.

    Parameters
    ----------
    {dataset}
    xlabel: str
        X-axis label.
    ylabel: str
        Y-axis label.
    title: str
        Title of the plot.
    {min_threshold}
    {ax}
    {kwargs_doc}

    {returns_ax}
    """
    df, n_pressent, n_total = dataset.record_counts(
        na_threshold=min_threshold,
        value_type="present"
    )
    prc_missing = n_pressent / n_total * 100

    return _plot_barplot(
        x_values=df["experiment"].tolist(),
        y_values=df["n_counts"].tolist(),
        group=df["condition"].tolist(),
        title=title + f" (~{round(prc_missing, 1)}% complete)",
        xlabel=xlabel,
        ylabel=ylabel,
        ax=ax,
        **kwargs
    )
