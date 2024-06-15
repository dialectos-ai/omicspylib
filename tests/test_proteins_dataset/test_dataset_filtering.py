from omicspylib import ProteinsDataset


def test_protein_dataset_filter_on_conditions(
        proteins_dataset: ProteinsDataset):
    """
    Test that you can filter in/out certain experimental
    conditions from a given dataset.
    """
    # setup
    c1 = 'c1'
    c2 = 'c2'
    c3 = 'c3'

    # action
    filt_dataset = proteins_dataset.filter(conditions=[c1, c2])

    # assertion
    assert c1 in filt_dataset.conditions
    assert c2 in filt_dataset.conditions
    assert c3 not in filt_dataset.conditions


def test_filtering_of_rows_with_low_within_group_frequency(
        proteins_dataset: ProteinsDataset):
    """
    For certain cases you might want to drop cases with low ids
    across experimental conditions. Test that you can filter them out.
    """
    # setup
    dset_before = proteins_dataset.filter(conditions=['c1'])

    # action
    dset_after = dset_before.filter(min_frequency=3, na_threshold=0.0)

    # assertion
    assert dset_before.n_records == 100
    assert dset_after.n_records < 100
