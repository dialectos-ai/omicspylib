"""Test proteins dataset plotting functions."""
import matplotlib.pyplot as plt

from omicspylib.plots import plot_density, plot_missing_values


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
