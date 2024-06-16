"""Test proteins dataset plotting functions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from omicspylib.plots import plot_density, plot_missing_values, plot_volcano


def test_intensity_plotting_as_density(proteins_dataset):
    # action
    ax = plot_density(proteins_dataset, log_transform=True)

    # assertion
    assert isinstance(ax, plt.Axes)


def test_missing_values_plotting_fn(proteins_dataset):
    # action
    ax = plot_missing_values(proteins_dataset)

    # assertion
    assert isinstance(ax, plt.Axes)


def test_volcano_plotting_fn():
    # setup
    pvals = np.random.rand(1000)
    fc_vals = 4 * np.random.rand(1000)
    data = pd.DataFrame({'p-values': pvals, 'fc_vals': fc_vals})

    # action
    ax = plot_volcano(data, pval_col='p-values', fc_col='fc_vals',
                      condition_a='a', condition_b='b')

    # assertion
    assert isinstance(ax, plt.Axes)