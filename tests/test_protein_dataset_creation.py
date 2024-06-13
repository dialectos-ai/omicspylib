"""Test proteins dataset creation flow."""
import pandas as pd
import pytest

from omicspylib import ProteinsDataset


def test_protein_dataset_creation_from_df():
    """
    Basic protein dataset constractor. Test that
    created object has expected attributes.
    """
    # setup
    data_df = pd.read_csv('tests/data/protein_dataset.tsv', sep='\t')
    config = {
        'id_col': 'protein_id',
        'conditions': {
            'c1': ['c1_rep1', 'c1_rep2', 'c1_rep3', 'c1_rep4', 'c1_rep5'],
            'c2': ['c2_rep1', 'c2_rep2', 'c2_rep3', 'c2_rep4', 'c2_rep5'],
            'c3': ['c3_rep1', 'c3_rep2', 'c3_rep3', 'c3_rep4', 'c3_rep5'],
        }
    }

    # action
    dataset = ProteinsDataset.from_df(data_df, **config)

    # assertion
    assert dataset.n_conditions == 3
    assert dataset.n_experiments == 15
    assert dataset.n_records == 100
    experimental_conditions = dataset.exp_conditions
    for condition_name, exp_names in config['conditions'].items():
        assert condition_name in experimental_conditions
        experiment_names = dataset.experiments(condition_name)
        for exp_name in exp_names:
            assert exp_name in experiment_names


@pytest.mark.parametrize(
    "rm_contaminants, rm_reverse, rm_modified, n_rows",
    [
        (False, False, False, 100),  # don't remove anything
        (True, False, False, 94),  # rm only contaminants
        (False, True, False, 96),  # rm only reverse
        (False, False, True, 97),  # rm only proteins identified by modified peptides
        (True, True, True, 87),  # rm all irrelevant
    ]
)
def test_protein_dataset_creation_from_mq_output(rm_contaminants, rm_reverse, rm_modified, n_rows):
    """
    You can create a dataset object directly from proteinGroups.txt file from
    MaxQuant. This is a wrapper around the `ProteinsDataset.from_df` method.
    Test that basic cleaning is done as expected during initialization.
    """
    # setup
    protein_groups_fp = "tests/data/protein_dataset_mq.tsv"
    data = pd.read_csv(protein_groups_fp, sep='\t')

    config = {
        'id_col': 'Majority protein IDs',
        'rename_id_col': 'protein_id',
        'conditions': {
            'c1': ['c1_rep1', 'c1_rep2', 'c1_rep3', 'c1_rep4', 'c1_rep5'],
            'c2': ['c2_rep1', 'c2_rep2', 'c2_rep3', 'c2_rep4', 'c2_rep5'],
            'c3': ['c3_rep1', 'c3_rep2', 'c3_rep3', 'c3_rep4', 'c3_rep5'],
        }
    }

    # action
    dset_from_fp = ProteinsDataset.from_maxquant(
        data=protein_groups_fp,
        rm_contaminants=rm_contaminants,
        rm_reverse=rm_reverse,
        rm_only_modified=rm_modified,
        **config)
    dset_from_df = ProteinsDataset.from_maxquant(
        data=data,
        rm_contaminants=rm_contaminants,
        rm_reverse=rm_reverse,
        rm_only_modified=rm_modified,
        **config)

    # assertion
    assert dset_from_fp.n_records == n_rows
    assert dset_from_df.n_records == n_rows
