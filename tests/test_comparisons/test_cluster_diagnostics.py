import numpy as np
import pandas as pd
import pytest
import scipy
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)

from omicspylib import ClusterDiagnostics, HierarchicallyClusteredHeatmap, ProteinsDataset
from omicspylib.analysis.clusters import HCHeatmapData, _silhouette_samples


@pytest.fixture
def hc_data(proteins_dataset: ProteinsDataset) -> HCHeatmapData:
    clustering = HierarchicallyClusteredHeatmap(n_row_clusters=6, n_col_clusters=3)
    return clustering.eval(data=proteins_dataset.to_table())


def test_linkages_are_returned(hc_data) -> None:
    n_rows, n_cols = hc_data.heatmap_inputs.shape
    assert hc_data.row_linkage.shape == (n_rows - 1, 4)
    assert hc_data.col_linkage.shape == (n_cols - 1, 4)


def test_item_tables(hc_data) -> None:
    result = ClusterDiagnostics().eval(hc_data)

    for items, groups, inputs, missing in [
        (
            result.records,
            hc_data.row_groups,
            hc_data.heatmap_inputs,
            hc_data.filtered_data.isna().mean(axis=1),
        ),
        (
            result.experiments,
            hc_data.col_groups,
            hc_data.heatmap_inputs.T,
            hc_data.filtered_data.isna().mean(axis=0),
        ),
    ]:
        assert items is not None
        assert list(items.columns) == ["group", "silhouette", "nearest group", "imputed fraction"]
        assert items.index.equals(inputs.index)
        assert items["group"].tolist() == groups
        np.testing.assert_allclose(items["silhouette"], np.asarray(silhouette_samples(inputs, groups), dtype=float))
        assert bool((items["nearest group"] != items["group"]).all())
        assert set(items["nearest group"]) <= set(groups)
        np.testing.assert_allclose(items["imputed fraction"], missing.loc[inputs.index])


def test_mean_silhouette_matches_heatmap_score(hc_data) -> None:
    result = ClusterDiagnostics().eval(hc_data)
    assert result.records is not None
    assert result.experiments is not None

    assert result.records["silhouette"].mean() == pytest.approx(hc_data.row_silhouette_score)
    assert result.experiments["silhouette"].mean() == pytest.approx(hc_data.col_silhouette_score)


def test_k_scan(hc_data) -> None:
    result = ClusterDiagnostics(k_range=range(2, 21)).eval(hc_data)
    n_cols = hc_data.heatmap_inputs.shape[1]
    row_k_scan, col_k_scan = result.row_k_scan, result.col_k_scan
    assert row_k_scan is not None
    assert col_k_scan is not None

    # rows: all k values are valid; columns: capped to n - 1
    assert row_k_scan.index.tolist() == list(range(2, 21))
    assert col_k_scan.index.tolist() == list(range(2, n_cols))
    assert list(row_k_scan.columns) == [
        "silhouette",
        "calinski-harabasz",
        "davies-bouldin",
        "merge height",
        "height gap",
        "min group size",
    ]

    x = hc_data.heatmap_inputs.to_numpy()
    k = 5
    labels = scipy.cluster.hierarchy.cut_tree(hc_data.row_linkage, n_clusters=[k]).ravel()
    row = row_k_scan.to_dict(orient="index")[k]

    def value(column: str) -> float:
        return float(row[column])

    assert value("silhouette") == pytest.approx(np.asarray(silhouette_samples(x, labels), dtype=float).mean())
    assert value("calinski-harabasz") == pytest.approx(calinski_harabasz_score(x, labels))
    assert value("davies-bouldin") == pytest.approx(davies_bouldin_score(x, labels))
    assert value("min group size") == np.bincount(labels).min()

    # cutting just below the merge height still gives at least k groups
    height = value("merge height")
    assert len(set(scipy.cluster.hierarchy.fcluster(hc_data.row_linkage, height * 0.9999, "distance"))) >= k
    assert value("height gap") >= 0  # average linkage is monotonic


def test_k_scan_matches_chosen_number_of_groups(hc_data) -> None:
    result = ClusterDiagnostics().eval(hc_data)
    assert result.row_k_scan is not None
    assert result.records is not None

    k_scan_silhouette = result.row_k_scan.to_dict(orient="index")[6]["silhouette"]
    assert k_scan_silhouette == pytest.approx(result.records["silhouette"].mean())


def test_k_scan_can_be_skipped(hc_data) -> None:
    result = ClusterDiagnostics(k_range=None).eval(hc_data)

    assert result.row_k_scan is None
    assert result.col_k_scan is None
    assert result.records is not None


def test_cophenetic_correlation(hc_data) -> None:
    result = ClusterDiagnostics().eval(hc_data)

    expected = scipy.cluster.hierarchy.cophenet(
        hc_data.row_linkage, scipy.spatial.distance.pdist(hc_data.heatmap_inputs.to_numpy())
    )[0]
    assert result.row_cophenetic_correlation == pytest.approx(expected)
    assert result.col_cophenetic_correlation is not None
    assert -1 <= result.col_cophenetic_correlation <= 1


def test_skipped_axis(proteins_dataset: ProteinsDataset) -> None:
    hc_data = HierarchicallyClusteredHeatmap(n_row_clusters=None, n_col_clusters=3).eval(
        proteins_dataset.to_table()
    )

    result = ClusterDiagnostics().eval(hc_data)

    assert result.records is None
    assert result.row_k_scan is None
    assert result.row_cophenetic_correlation is None
    assert result.experiments is not None


def test_missing_linkage_raises(hc_data) -> None:
    hc_data.row_linkage = None

    with pytest.raises(ValueError, match="row_linkage"):
        ClusterDiagnostics().eval(hc_data)


def test_silhouette_chunking_and_singletons() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    labels = np.repeat([0, 1, 2], 10)
    labels[0] = 3  # singleton group

    (full, nearest_full), = _silhouette_samples(x, [labels], chunk_size=1024)
    (chunked, nearest_chunked), = _silhouette_samples(x, [labels], chunk_size=7)

    np.testing.assert_allclose(full, np.asarray(silhouette_samples(x, labels), dtype=float))
    np.testing.assert_allclose(chunked, full)
    assert list(nearest_chunked) == list(nearest_full)
    assert full[0] == 0.0


def test_silhouette_single_group() -> None:
    x = np.arange(10, dtype=float).reshape(5, 2)

    (silhouette, nearest), = _silhouette_samples(x, [np.zeros(5, dtype=int)])

    assert np.isnan(silhouette).all()
    assert pd.isna(nearest).all()
