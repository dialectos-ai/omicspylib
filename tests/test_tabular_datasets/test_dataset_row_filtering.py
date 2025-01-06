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
        assert c1 in filt_dataset.condition_names
        assert c2 in filt_dataset.condition_names
        assert c3 not in filt_dataset.condition_names


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


@pytest.mark.parametrize(
    "dset_name,target_ids",
    [
        ('proteins', ['p1', 'p2', 'p3']),
        ('peptides', ['pept1', 'pept2', 'pept3']),
    ]
)
def test_filter_dataset_based_on_primary_id(
        proteins_dataset: ProteinsDataset,
        peptides_dataset: PeptidesDataset,
        dset_name, target_ids):
    """
    Test that you can keep in a dataset only the target ids.
    """
    # setup
    dsets = {
        'proteins': proteins_dataset,
        'peptides': peptides_dataset
    }
    dataset = dsets[dset_name]

    # action
    filtered_dset = dataset.filter(ids=target_ids)

    # assertion
    filt_dset_ids = filtered_dset.to_table().index.tolist()
    for tid in target_ids:
        assert tid in filt_dset_ids
    for fid in filt_dset_ids:
        assert fid in target_ids


def test_filter_peptides_dataset_based_on_protein_id(
        peptides_dataset: PeptidesDataset,):
    """
    Test that you can indirectly filter peptide records,
    based on the associated protein id.
    """
    # setup
    target_ids = ['prot1', 'prot2', 'prot3']

    # action
    filtered_dset = peptides_dataset.filter(protein_ids=target_ids)

    # assertion
    prot_ids = filtered_dset.to_proteins().to_table().index.tolist()
    for tid in target_ids:
        assert tid in prot_ids
    assert len(prot_ids) == len(target_ids)


def test_drop_peptides_dataset_records_based_on_protein_id(
        peptides_dataset: PeptidesDataset,):
    """
    Test that you can indirectly filter peptide records,
    based on the associated protein id.
    """
    # setup
    target_ids = ['prot1', 'prot2', 'prot3']

    # action
    filtered_dset = peptides_dataset.drop(protein_ids=target_ids)

    # assertion
    prot_ids = filtered_dset.to_proteins().to_table().index.tolist()
    for tid in target_ids:
        assert tid not in prot_ids


@pytest.mark.parametrize(
    "dset_name,target_ids",
    [
        ('proteins', ['p1', 'p2', 'p3']),
        ('peptides', ['pept1', 'pept2', 'pept3']),
    ]
)
def test_you_can_drop_specific_records_from_the_dataset_based_on_their_id(
        proteins_dataset: ProteinsDataset,
        peptides_dataset: PeptidesDataset,
        dset_name, target_ids):
    """
    Test that you can keep in a dataset only the target ids.
    """
    # setup
    dsets = {
        'proteins': proteins_dataset,
        'peptides': peptides_dataset
    }
    dataset = dsets[dset_name]

    # action
    filtered_dset = dataset.drop(ids=target_ids)

    # assertion
    filt_dset_ids = filtered_dset.to_table().index.tolist()
    dset_ids = dataset.to_table().index.tolist()
    for tid in target_ids:
        assert tid not in filt_dset_ids
        assert tid in dset_ids
