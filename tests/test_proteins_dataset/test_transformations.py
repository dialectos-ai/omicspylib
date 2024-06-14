"""Test protein dataset transformations."""
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