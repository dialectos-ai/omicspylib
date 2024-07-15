import pytest

from omicspylib import ProteinsDataset, PeptidesDataset


def test_protein_dataset_filter_on_conditions(
        proteins_dataset: ProteinsDataset,
        peptides_dataset: PeptidesDataset):
    """
    Test that you can filter in/out certain experimental
    conditions from a given dataset.
    """
    # setup
    c1 = 'c1'
    c2 = 'c2'
    c3 = 'c3'
    datasets = [proteins_dataset, peptides_dataset]

    # action
    for dataset in datasets:
        filt_dataset = dataset.filter(cond=[c1, c2])

        # assertion
        assert c1 in filt_dataset.conditions
        assert c2 in filt_dataset.conditions
        assert c3 not in filt_dataset.conditions


@pytest.mark.parametrize(
    "dset_name,conditions,min_f,n_before,n_after",
    [
        ('proteins', ['c1'], 3, 100, 100),
        ('peptides', ['c1'], 3, 1000, 1000),
    ]
)
def test_filtering_of_rows_with_low_within_group_frequency(
        proteins_dataset: ProteinsDataset,
        peptides_dataset: PeptidesDataset,
        dset_name,conditions,min_f,n_before,n_after):
    """
    For certain cases, you might want to drop cases with low ids
    across experimental conditions. Test that you can filter them out.
    """
    # setup
    dsets = {
        'proteins': proteins_dataset,
        'peptides': peptides_dataset
    }
    dataset = dsets[dset_name]
    dset_before = dataset.filter(cond=conditions)

    # action
    dset_after = dset_before.filter(min_frequency=min_f, na_threshold=0.0)

    # assertion
    assert dset_before.n_records == n_before
    assert dset_after.n_records < n_after
