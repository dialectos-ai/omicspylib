import pytest

from omicspylib import ProteinsDataset, PeptidesDataset


@pytest.mark.parametrize(
    "dset_name,drop_exp,drop_cond",
    [
        ('proteins', 'c1_rep1', None),
        ('proteins', ['c1_rep1', 'c2_rep2'], None),
        ('proteins', None, 'c1'),
        ('proteins', None, ['c1', 'c2']),
        ('peptides', 'c1_rep1', None),
        ('peptides', ['c1_rep1', 'c2_rep2'], None),
        ('peptides', None, 'c1'),
        ('peptides', None, ['c1', 'c2']),
    ]

)
def test_exp_dropping_in_dataset(
        proteins_dataset: ProteinsDataset,
        peptides_dataset: PeptidesDataset, dset_name, drop_exp, drop_cond):
    """
    Test that you can drop specific experiment(s) from a dataset.
    """
    # action
    dsets = {
        'proteins': proteins_dataset,
        'peptides': peptides_dataset
    }
    dataset = dsets[dset_name]

    new_dataset = dataset.drop(exp=drop_exp, cond=drop_cond)

    # assertion
    if drop_cond is None:
        exp_before = dataset.experiments()
        exp_after = new_dataset.experiments()

        if isinstance(drop_exp, str):
            drop_exp = [drop_exp]

        for exp in drop_exp:
            assert exp in exp_before
            assert exp not in exp_after
    else:
        cond_before = dataset.conditions
        cond_after = new_dataset.conditions

        if isinstance(drop_cond, str):
            drop_cond = [drop_cond]

        for cond in drop_cond:
            # assertion
            assert cond in cond_before
            assert cond not in cond_after
