"""Fixtures shared across tests"""
import pandas as pd
import pytest

from omicspylib import ProteinsDataset


@pytest.fixture
def proteins_dataset():
    """Create a basic proteins dataset object."""
    data_df = pd.read_csv('tests/data/protein_dataset.tsv', sep='\t')
    config = {
        'id_col': 'protein_id',
        'conditions': {
            'c1': ['c1_rep1', 'c1_rep2', 'c1_rep3', 'c1_rep4', 'c1_rep5'],
            'c2': ['c2_rep1', 'c2_rep2', 'c2_rep3', 'c2_rep4', 'c2_rep5'],
            'c3': ['c3_rep1', 'c3_rep2', 'c3_rep3', 'c3_rep4', 'c3_rep5'],
        }
    }

    return ProteinsDataset.from_df(data_df, **config)
