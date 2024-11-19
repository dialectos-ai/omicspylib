import pandas as pd
from omicspylib.datasets.peptides import PeptidesDataset

import omicspylib as opl

from omicspylib import ProteinsDataset


def test_protein_dataset_concatenation():
    """
    You can have to datasets with different experimental conditions.
    Text that you can join them in one object.
    """
    # setup
    """Create a basic proteins dataset object."""
    data_df = pd.read_csv('tests/data/protein_dataset.tsv', sep='\t')
    d1_config = {
        'id_col': 'protein_id',
        'conditions': {'c1': ['c1_rep1', 'c1_rep2', 'c1_rep3', 'c1_rep4', 'c1_rep5'],}}
    d2_config = {
        'id_col': 'protein_id',
        'conditions': {'c2': ['c2_rep1', 'c2_rep2', 'c2_rep3', 'c2_rep4', 'c2_rep5'],}}
    d3_config = {
        'id_col': 'protein_id',
        'conditions': {'c3': ['c3_rep1', 'c3_rep2', 'c3_rep3', 'c3_rep4', 'c3_rep5'],}}
    d1 = ProteinsDataset.from_df(data_df, **d1_config)
    d2 = ProteinsDataset.from_df(data_df, **d2_config)
    d3 = ProteinsDataset.from_df(data_df, **d3_config)

    # action
    joined_dset = opl.concat([d1, d2, d3])

    # assertion
    assert isinstance(joined_dset, ProteinsDataset)
    conditions = joined_dset.condition_names
    for exp_cond in ['c1', 'c2', 'c3']:
        assert exp_cond in conditions


def test_peptide_dataset_concatenation():
    """
    You can have to datasets with different experimental conditions.
    Text that you can join them in one object.
    """
    # setup
    """Create a basic proteins dataset object."""
    data_df = pd.read_csv('tests/data/peptides_dataset.tsv', sep='\t')
    d1_config = {
        'id_col': 'peptide_id',
        'protein_id_col': 'protein_id',
        'conditions': {'c1': ['c1_rep1', 'c1_rep2', 'c1_rep3', 'c1_rep4', 'c1_rep5'],}}
    d2_config = {
        'id_col': 'peptide_id',
        'protein_id_col': 'protein_id',
        'conditions': {'c2': ['c2_rep1', 'c2_rep2', 'c2_rep3', 'c2_rep4', 'c2_rep5'],}}
    d3_config = {
        'id_col': 'peptide_id',
        'protein_id_col': 'protein_id',
        'conditions': {'c3': ['c3_rep1', 'c3_rep2', 'c3_rep3', 'c3_rep4', 'c3_rep5'],}}
    d1 = PeptidesDataset.from_df(data_df, **d1_config)
    d2 = PeptidesDataset.from_df(data_df, **d2_config)
    d3 = PeptidesDataset.from_df(data_df, **d3_config)

    # action
    joined_dset = opl.concat([d1, d2, d3])

    # assertion
    assert isinstance(joined_dset, PeptidesDataset)
    conditions = joined_dset.condition_names
    for exp_cond in ['c1', 'c2', 'c3']:
        assert exp_cond in conditions