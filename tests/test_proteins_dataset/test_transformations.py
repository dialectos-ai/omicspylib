"""Test protein dataset transformations."""
import numpy as np
import pandas as pd

from omicspylib import ProteinsDataset


def test_proteins_dataset_conversion_to_table(proteins_dataset: ProteinsDataset):
    """
    For various tasks you need to merge all conditions
    of a proteins dataset back to one table. Test that.
    """
    # action
    table = proteins_dataset.to_table()

    # assertion
    assert isinstance(table, pd.DataFrame)
    assert table.shape[0] == proteins_dataset.n_records
    assert table.shape[1] == proteins_dataset.n_experiments


def test_log2_transformation(proteins_dataset: ProteinsDataset):
    # action
    log2_dset = proteins_dataset.log2_transform().to_table()

    # assert
    mean_raw_int = np.mean(proteins_dataset.to_table().values.reshape(-1))
    mean_log2_int = np.mean(log2_dset.values.reshape(-1))
    assert np.isclose(mean_raw_int, 1_567_714, 1_000_000)
    assert np.isclose(mean_log2_int, 25, 10)

