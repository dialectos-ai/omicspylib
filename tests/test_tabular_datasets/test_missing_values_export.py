"""Exporting the data with missing values made explicit."""

import numpy as np
import pandas as pd
import pytest

from omicspylib import PeptidesDataset, ProteinsDataset


@pytest.fixture
def small_dataset() -> ProteinsDataset:
    data = pd.DataFrame({
        "protein_id": ["p1", "p2", "p3"],
        "a1": [10.0, 0.0, np.nan],
        "a2": [12.0, 5.0, 0.0],
        "b1": [0.0, 7.0, 3.0],
        "b2": [1.0, 8.0, 4.0],
    })
    return ProteinsDataset.from_df(
        data, id_col="protein_id", conditions={"a": ["a1", "a2"], "b": ["b1", "b2"]}
    )


def test_to_table_default_is_unchanged(small_dataset: ProteinsDataset) -> None:
    table = small_dataset.to_table()
    assert table.loc["p2", "a1"] == 0.0
    assert table.loc["p1", "b1"] == 0.0


def test_to_table_mask_missing(small_dataset: ProteinsDataset) -> None:
    table = small_dataset.to_table(mask_missing=True)
    assert np.isnan(table.loc["p2", "a1"])
    assert np.isnan(table.loc["p1", "b1"])
    assert np.isnan(table.loc["p3", "a1"])
    assert table.loc["p1", "a1"] == 10.0
    # pandas now skips missing values instead of counting zeros
    assert table.loc["p1", ["b1", "b2"]].mean() == 1.0


def test_to_table_mask_missing_with_threshold(small_dataset: ProteinsDataset) -> None:
    table = small_dataset.to_table(mask_missing=True, na_threshold=3.0)
    assert np.isnan(table.loc["p3", "b1"])  # 3.0 <= threshold
    assert table.loc["p3", "b2"] == 4.0


def test_to_table_mask_missing_does_not_modify_dataset(small_dataset: ProteinsDataset) -> None:
    small_dataset.to_table(mask_missing=True)
    assert small_dataset.to_table().loc["p2", "a1"] == 0.0


def test_detection_matrix(small_dataset: ProteinsDataset) -> None:
    detected = small_dataset.detection_matrix()
    expected = pd.DataFrame(
        {
            "a1": [True, False, False],
            "a2": [True, True, False],
            "b1": [False, True, True],
            "b2": [True, True, True],
        },
        index=pd.Index(["p1", "p2", "p3"], name="protein_id"),
    )
    pd.testing.assert_frame_equal(detected, expected, check_names=False)


def test_detection_matrix_matches_frequency(proteins_dataset: ProteinsDataset) -> None:
    detected = proteins_dataset.detection_matrix(join_method="inner")
    freq = proteins_dataset.frequency(join_method="inner")
    for cond in proteins_dataset.condition_names:
        counts = detected[proteins_dataset.experiment_names(cond)].sum(axis=1)
        np.testing.assert_array_equal(
            counts.to_numpy(), freq.loc[counts.index, f"frequency_{cond}"].to_numpy()
        )


def test_detection_matrix_absent_records_not_detected(proteins_dataset: ProteinsDataset) -> None:
    # after per-condition filtering, records are absent from some conditions
    filtered = proteins_dataset.filter(min_frequency=5)
    table = filtered.to_table(join_method="outer")
    detected = filtered.detection_matrix(join_method="outer")
    assert detected.shape == table.shape
    assert not detected[table.isna()].any().any()


def test_counts_respect_na_threshold() -> None:
    data = pd.DataFrame({
        "peptide": ["s1", "s2", "s3"],
        "protein_id": ["p1", "p1", "p1"],
        "a1": [10.0, 2.0, 0.0],
        "a2": [10.0, 10.0, 10.0],
    })
    peptides = PeptidesDataset.from_df(
        data, id_col="peptide", conditions={"a": ["a1", "a2"]}, protein_id_col="protein_id"
    )
    default = peptides.to_proteins(agg_method="counts").to_table()
    assert default.loc["p1", "a1"] == 2
    thresholded = peptides.to_proteins(agg_method="counts", na_threshold=5.0).to_table()
    assert thresholded.loc["p1", "a1"] == 1
    assert thresholded.loc["p1", "a2"] == 3
