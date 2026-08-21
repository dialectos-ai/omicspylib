from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from omicspylib.datasets.abc import TabularDataset


def plot_record_across_experiments(
    dataset: TabularDataset,
    plot_type: Literal["jitter", "bar"],
    record_id: str,
    log_transform: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Plot values for a specific record across experiments as either a jitter plot
    (grouped by condition) or a bar plot (per experiment).

    Parameters
    ----------
    dataset : TabularDataset
        Dataset object containing quantitative values and experiment conditions.
    plot_type : Literal["jitter", "bar"]
        Type of plot to generate: "jitter" (boxplot + stripplot grouped by condition)
        or "bar" (barplot across experiments).
    record_id : str
        ID of the record (protein/gene) to plot.
    log_transform : bool, default=False
        If True, transforms values to a log2 scale.
    xlabel : str | None, default=None
        Label for the x-axis. If None, defaults to "Condition" for jitter and "Experiment" for bar.
    ylabel : str | None, default=None
        Label for the y-axis. If None, defaults based on log_transform.
    title : str | None, default=None
        Title of the plot. If None, defaults to "Record: {record_id}".
    ax : Axes | None, default=None
        Matplotlib Axes object to draw on. If None, a new figure and axes are created.
    **kwargs : dict, optional
        Additional keyword arguments:

        * show_experiment_name : bool, default=False
            If True, displays experiment names on the jitter plot.
        * text_annotation_size : float, default=5
            Font size for text annotations on jitter plot points.
        * text_rotation : float, default=0
            Rotation angle in degrees for text annotations.
        * text_round_digits : int | None, default=None
            If specified, rounds numeric text annotations to this number of decimal places.
    Returns
    -------
    Axes
        Matplotlib Axes object containing the plot.
    """
    text_annotation_size = kwargs.get("text_annotation_size", 6)
    text_rotation = kwargs.get("text_rotation", 0)
    text_round_digits = kwargs.get("text_round_digits", None)

    tabular_df = dataset.to_table()
    if record_id not in tabular_df.index:
        raise ValueError(f"Record ID '{record_id}' not found in dataset.")

    # Get experiment-to-condition mapping
    counts_df, _, _ = dataset.record_counts()
    exp_to_cond = dict(zip(counts_df["experiment"], counts_df["condition"], strict=True))

    # Prepare single-record DataFrame
    rec_series = tabular_df.loc[record_id]
    plot_df = pd.DataFrame(
        {
            "experiment": rec_series.index,
            "value": rec_series.values,
        }
    )
    plot_df["condition"] = plot_df["experiment"].map(exp_to_cond)
    plot_df = plot_df.loc[plot_df["value"] > 0].copy()

    if log_transform:
        plot_df["value"] = np.log2(plot_df["value"])

    if ax is None:
        _, ax = plt.subplots()

    unique_conditions = list(plot_df["condition"].unique())
    colors = plt.get_cmap("tab20").colors  # pyright: ignore
    cmap = {cond: colors[i % len(colors)] for i, cond in enumerate(unique_conditions)}

    default_ylabel = "Log2 Abundance" if log_transform else "Abundance / Intensity"

    if plot_type == "jitter":
        # 1. Boxplot grouped by condition
        sns.boxplot(
            data=plot_df,
            x="condition",
            y="value",
            hue="condition",
            palette=cmap,
            legend=False,
            showfliers=False,
            width=0.4,
            boxprops=dict(alpha=0.6),  # noqa: C408
            ax=ax,
        )

        # 2. Stripplot for jittered individual data points
        sns.stripplot(
            data=plot_df,
            x="condition",
            y="value",
            hue="condition",
            palette=cmap,
            legend=False,
            size=8,
            jitter=0.2,
            edgecolor="black",
            linewidth=1,
            ax=ax,
        )
        if kwargs.get("show_experiment_names", False):
            points = np.vstack(
                [coll.get_offsets() for coll in ax.collections if np.asarray(coll.get_offsets()).size > 0]
            )
            padding = 0.1
            for (x, y), (_, row) in zip(points, plot_df.iterrows(), strict=True):
                ax.text(
                    x + padding,
                    y,
                    row["experiment"],
                    va="center",
                    ha="left",
                    fontsize=text_annotation_size,
                )
        ax.set_xlabel(xlabel if xlabel is not None else "Condition")

    elif plot_type == "bar":
        exp_colors = [cmap.get(cond, colors[0]) for cond in plot_df["condition"]]
        bars = ax.bar(x=plot_df["experiment"], height=plot_df["value"], color=exp_colors)

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(plot_df["experiment"], rotation=45, ha="right")
        ax.bar_label(bars, fontsize=text_annotation_size)
        ax.set_xlabel(xlabel if xlabel is not None else "Experiment")

    else:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Must be 'jitter' or 'bar'.")

    # Apply rotation and rounding formatting to all text annotations
    for txt in ax.texts:
        if text_rotation != 0:
            txt.set_rotation(text_rotation)
        if text_round_digits is not None:
            try:
                val = float(txt.get_text())
                txt.set_text(f"{val:.{text_round_digits}f}")
            except ValueError:
                pass

    ax.set_title(title if title is not None else f"Record: {record_id}", fontweight="bold")
    ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return ax
