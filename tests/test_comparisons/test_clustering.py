import pytest
from seaborn.matrix import ClusterGrid

from omicspylib.analysis.clusters import HierarchicallyClusteredHeatmap


def test_clustering_success_case(proteins_dataset):
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

@pytest.mark.parametrize("fillna_method,min_frequency,n_rows", [
    ('min', 0, 100),
    ('min', 5, 97),
    ('drop', 0, 2)
])
def test_clustering_filtering_on_missing_values(
        proteins_dataset, fillna_method, min_frequency, n_rows):
    # setup
    clustering = HierarchicallyClusteredHeatmap(
        fillna_method=fillna_method,
        min_frequency=min_frequency)
    data_in = proteins_dataset.to_table()

    # action
    result = clustering.eval(data=data_in)

    # assertion
    assert result.filtered_data.shape[0] == n_rows
    assert len(result.row_groups) == n_rows
    assert len(result.col_groups) == data_in.shape[1]


@pytest.mark.parametrize("n_row_clusters,n_col_clusters", [
    (12, 3),
    (12, None),
    (12, 0),
    (None, 3),
    (0, 3)
])
def test_row_col_clustering_options(
        proteins_dataset, n_row_clusters, n_col_clusters):
    # setup
    clustering = HierarchicallyClusteredHeatmap(n_row_clusters=n_row_clusters,
        n_col_clusters=n_col_clusters)
    data_in = proteins_dataset.to_table()

    # action
    result = clustering.eval(data=data_in)

    # assertion
    if n_row_clusters is None or n_row_clusters == 0:
        assert result.row_groups is None
    else:
        assert len(list(set(result.row_groups))) == n_row_clusters
    if n_col_clusters is None or n_col_clusters == 0:
        assert result.col_groups is None
    else:
        assert len(list(set(result.col_groups))) == n_col_clusters
