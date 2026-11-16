import pytest
from seaborn.matrix import ClusterGrid

from omicspylib import ProteinsDataset
from omicspylib.analysis.clusters import HierarchicallyClusteredHeatmap


def test_clustering_success_case(proteins_dataset: ProteinsDataset) -> None:
    # setup
    comparison = HierarchicallyClusteredHeatmap()
    data_in = proteins_dataset.to_table()

    # action
    result = comparison.eval(data=data_in)

    # assertion
    assert result.filtered_data.shape == data_in.shape
    assert isinstance(result.g, ClusterGrid)
    assert isinstance(result.row_groups, list)
    assert isinstance(result.col_groups, list)


@pytest.mark.parametrize(
    ("fillna_method", "min_frequency", "n_rows"),
    [("min", 0, 100), ("min", 5, 98), ("drop", 0, 2)],
)
def test_clustering_filtering_on_missing_values(
    proteins_dataset: ProteinsDataset,
    fillna_method,
    min_frequency,
    n_rows,
) -> None:
    # setup
    clustering = HierarchicallyClusteredHeatmap(
        fillna_method=fillna_method, min_frequency=min_frequency
    )
    data_in = proteins_dataset.to_table()

    # action
    result = clustering.eval(data=data_in)

    # assertion
    assert result.filtered_data.shape[0] == n_rows
    assert result.row_groups is not None
    assert len(result.row_groups) == n_rows
    assert result.col_groups is not None
    assert len(result.col_groups) == data_in.shape[1]


@pytest.mark.parametrize(
    ("n_row_clusters", "n_col_clusters"), [(12, 3), (12, None), (12, 0), (None, 3), (0, 3)]
)
def test_row_col_clustering_options(
    proteins_dataset: ProteinsDataset,
    n_row_clusters: int | None,
    n_col_clusters: int | None,
) -> None:
    # setup
    clustering = HierarchicallyClusteredHeatmap(
        n_row_clusters=n_row_clusters, n_col_clusters=n_col_clusters
    )
    data_in = proteins_dataset.to_table()

    # action
    result = clustering.eval(data=data_in)

    # assertion
    if n_row_clusters is None or n_row_clusters == 0:
        assert result.row_groups is None
    else:
        assert result.row_groups is not None
        assert len(list(set(result.row_groups))) == n_row_clusters
    if n_col_clusters is None or n_col_clusters == 0:
        assert result.col_groups is None
    else:
        assert result.col_groups is not None
        assert len(list(set(result.col_groups))) == n_col_clusters


@pytest.mark.parametrize(("min_frequency", "kept"), [(2, True), (3, False)])
def test_clustering_min_frequency_counts_all_columns(min_frequency, kept) -> None:
    """A row with exactly N valid values is kept with ``min_frequency=N``."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.uniform(100, 1000, (6, 6)), columns=[f"e{i}" for i in range(6)], index=[f"p{i}" for i in range(6)]
    )
    data.loc["p0", ["e2", "e3", "e4", "e5"]] = 0  # p0 has exactly 2 valid values
    clustering = HierarchicallyClusteredHeatmap(min_frequency=min_frequency, n_row_clusters=2, n_col_clusters=2)

    result = clustering.eval(data=data)

    assert ("p0" in result.filtered_data.index) is kept


def test_row_annotations_aligned_on_index(proteins_dataset: ProteinsDataset) -> None:
    """Labels are matched on the index; filtered or unlabelled rows are handled."""
    import pandas as pd

    data_in = proteins_dataset.to_table()
    labels = pd.Series(
        ["a", "b"] * (len(data_in) // 2) + ["a"] * (len(data_in) % 2),
        index=data_in.index,
    )
    labels = labels.drop(labels.index[0])  # one row without a label
    clustering = HierarchicallyClusteredHeatmap(min_frequency=5, n_col_clusters=None)

    result = clustering.eval(
        data=data_in,
        row_annotations={"fraction": labels},
        row_annotation_palettes={"fraction": {"a": "black"}},
    )

    row_colors = result.g.row_colors
    assert row_colors is not None
    assert len(row_colors) == 2  # row groups + annotation track
    track = list(row_colors[1])
    assert len(track) == result.filtered_data.shape[0]
    expected = labels.reindex(result.filtered_data.index)
    for color, lbl in zip(track, expected, strict=True):
        if pd.isna(lbl):
            assert color == (1.0, 1.0, 1.0)
        elif lbl == "a":
            assert color == (0.0, 0.0, 0.0)


def test_row_annotations_reject_non_series(proteins_dataset: ProteinsDataset) -> None:
    clustering = HierarchicallyClusteredHeatmap()
    data_in = proteins_dataset.to_table()

    with pytest.raises(TypeError):
        clustering.eval(data=data_in, row_annotations={"x": ["a"] * len(data_in)})  # pyright: ignore


def test_rows_only_layout_reclaims_column_band(proteins_dataset: ProteinsDataset) -> None:
    result = HierarchicallyClusteredHeatmap(n_col_clusters=None).eval(
        data=proteins_dataset.to_table()
    )

    g = result.g
    assert not g.ax_col_dendrogram.get_visible()
    top = g.ax_col_dendrogram.get_position().y1
    assert g.ax_heatmap.get_position().y1 == pytest.approx(top)
    # colour bar sits left of the row dendrogram, labelled low/high
    assert g.ax_cbar is not None
    assert g.ax_cbar.get_position().x1 <= g.ax_row_dendrogram.get_position().x0
    labels = [t.get_text() for t in g.ax_cbar.get_yticklabels()]
    assert labels == ["low", "high"]
