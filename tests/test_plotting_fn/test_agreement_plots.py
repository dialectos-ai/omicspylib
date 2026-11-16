"""Test plots of the agreement between experiments."""
from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from omicspylib.datasets.proteins import ProteinsDataset
from omicspylib.plots import plot_correlation_heatmap, plot_experiment_pair


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_correlation_heatmap(proteins_dataset: ProteinsDataset, method) -> None:
    ax = plot_correlation_heatmap(proteins_dataset, method=method)

    assert isinstance(ax, Axes)
    matrix = np.asarray(ax.collections[0].get_array())
    n = proteins_dataset.n_experiments
    assert matrix.size == n * n
    assert ax.get_title() == f"{method.capitalize()} correlation"


def test_correlation_heatmap_matches_correlation(proteins_dataset) -> None:
    ax = plot_correlation_heatmap(proteins_dataset, show_values=False)

    n = proteins_dataset.n_experiments
    matrix = np.asarray(ax.collections[0].get_array()).reshape(n, n)
    corr = proteins_dataset.correlation(pairs="all")
    names = proteins_dataset.experiment_names()
    first = corr.iloc[0]
    i, j = names.index(first["experiment_a"]), names.index(first["experiment_b"])
    assert matrix[i, j] == pytest.approx(first["coefficient"])
    assert matrix[j, i] == pytest.approx(first["coefficient"])
    assert np.allclose(np.diag(matrix), 1.0)


def test_correlation_heatmap_hides_values_for_many_experiments(proteins_dataset) -> None:
    many = plot_correlation_heatmap(proteins_dataset)  # 15 experiments
    few = plot_correlation_heatmap(proteins_dataset.filter(conditions=["c1", "c2"]))

    assert len(many.texts) == 0
    assert len(few.texts) == 10 * 10


@pytest.mark.parametrize("plot_type", ["scatter", "bland_altman"])
def test_experiment_pair(proteins_dataset: ProteinsDataset, plot_type) -> None:
    ax = plot_experiment_pair(proteins_dataset, "c1_rep1", "c1_rep2", plot_type=plot_type)

    assert isinstance(ax, Axes)
    n_points = len(np.asarray(ax.collections[0].get_offsets()))
    n_complete = len(
        proteins_dataset.to_table(mask_missing=True)[["c1_rep1", "c1_rep2"]].dropna()
    )
    assert n_points == n_complete
    assert ax.get_title() == "c1_rep1 vs c1_rep2"


def test_bland_altman_plot_draws_bias_and_limits(proteins_dataset) -> None:
    ax = plot_experiment_pair(
        proteins_dataset, "c1_rep1", "c1_rep2", plot_type="bland_altman"
    )
    stats = proteins_dataset.bland_altman(pairs=[("c1_rep1", "c1_rep2")]).iloc[0]

    line_levels = sorted(float(np.asarray(line.get_ydata())[0]) for line in ax.get_lines())
    assert line_levels == pytest.approx(
        sorted([stats["loa_lower"], stats["bias"], stats["loa_upper"]])
    )


def test_experiment_pair_invalid_plot_type(proteins_dataset) -> None:
    with pytest.raises(ValueError, match="plot_type"):
        plot_experiment_pair(proteins_dataset, "c1_rep1", "c1_rep2", plot_type="hexbin")  # type: ignore[arg-type]
