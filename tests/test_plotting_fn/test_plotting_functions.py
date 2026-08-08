"""Test protein dataset plotting functions."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from omicspylib.datasets.proteins import ProteinsDataset
from omicspylib.plots import plot_density, plot_missing_values, plot_venn2, plot_volcano
from omicspylib.plots.frequency_barplot import plot_record_frequency


def test_intensity_plotting_as_density(proteins_dataset: ProteinsDataset) -> None:
    # action
    ax = plot_density(proteins_dataset, log_transform=True)

    # assertion
    assert isinstance(ax, Axes)


def test_missing_values_plotting_fn(proteins_dataset: ProteinsDataset) -> None:
    # action
    ax = plot_missing_values(proteins_dataset)

    # assertion
    assert isinstance(ax, Axes)
    plt.savefig("test_missing_values_plotting_fn.png")


def test_record_frequency_plotting_fn(proteins_dataset: ProteinsDataset) -> None:
    """Test that you can plot the number of records per experiment
    (e.g., number of proteins per run)."""
    # action
    ax = plot_record_frequency(proteins_dataset)

    # assertion
    assert isinstance(ax, Axes)
    plt.savefig("test_record_frequency_plotting_fn.png")


def test_volcano_plotting_fn() -> None:
    # setup
    pvals = np.random.rand(1000)
    fc_vals = 4 * np.random.rand(1000)
    data = pd.DataFrame({"p-values": pvals, "fc_vals": fc_vals})

    # action
    ax = plot_volcano(
        data, pval_col="p-values", fc_col="fc_vals", condition_a="a", condition_b="b"
    )

    # assertion
    assert isinstance(ax, Axes)


def test_plot_venn2() -> None:
    # setup
    unique_a = ["unique c1" for _ in range(50)]
    unique_b = ["unique c2" for _ in range(100)]
    common = ["common" for _ in range(250)]
    none = ["none" for _ in range(40)]

    fclasses = unique_a + unique_b + common + none
    df = pd.DataFrame({"frequency_class": fclasses})

    # action
    ax = plot_venn2(df, condition_a="c1", condition_b="c2")

    # assert
    assert isinstance(ax, Axes)
