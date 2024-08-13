import numpy as np
import pytest

from omicspylib import ProteinsDataset, PeptidesDataset


@pytest.mark.parametrize(
    "dset_name,row_idx,impute_method,targ_cols,targ_values,a_tol,kwargs",
    [
        ('proteins', 'p5', "fixed", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [100, 100, 100], 0.1, {'value': 100}),
        ('proteins', 'p5', "fixed", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [100, 100, 100], 1.5, {'value': 100, 'random_noise': True}),
        ('proteins', 'p5', "global min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.046, 20.046, 20.046], 0.1, {}),
        ('proteins', 'p5', "global min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.046, 15.046, 15.046], 0.1, {'shift': 5}),
        ('proteins', 'p5', "global min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.046, 15.046, 15.046], 2.5, {'shift': 5, 'random_noise': True}),
        ('proteins', 'p5', "global mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.82, 20.82, 20.82], 0.1, {}),
        ('proteins', 'p5', "global median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.83, 20.83, 20.83], 0.1, {}),
        ('proteins', 'p5', "global row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.22, 20.22, 20.22], 0.1, {}),
        ('proteins', 'p5', "global row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.22, 15.22, 15.22], 0.1, {'shift': 5}),
        ('proteins', 'p5', "global row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.22, 15.22, 15.22], 2.5, {'shift': 5, 'random_noise': True}),
        ('proteins', 'p5', "global row mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.77, 20.77, 20.77], 0.1, {}),
        ('proteins', 'p5', "global row median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.74, 20.74, 20.74], 0.1, {}),
        ('proteins', 'p5', "group row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.95, 20.95, 20.22], 0.1, {}),
        ('proteins', 'p5', "group row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.95, 15.95, 15.22], 0.1, {'shift': 5}),
        ('proteins', 'p5', "group row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.95, 15.95, 15.22], 2.5, {'shift': 5, 'random_noise': True}),
        ('proteins', 'p5', "group row mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [21.03, 21.03, 20.52], 0.1, {}),
        ('proteins', 'p5', "group row mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [16.03, 16.03, 15.52], 0.1, {'shift': 5}),
        ('proteins', 'p5', "group row mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [16.03, 16.03, 15.52], 2.5, {'shift': 5, 'random_noise': True}),
        ('proteins', 'p5', "group row median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [21.03, 21.03, 20.56], 0.1, {}),
        ('proteins', 'p5', "group row median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [16.03, 16.03, 15.56], 0.1, {'shift': 5}),
        ('proteins', 'p5', "group row median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [16.03, 16.03, 15.56], 2.5, {'shift': 5, 'random_noise': True}),
        ('peptides', 'pept424', "fixed", ['c1_rep2', 'c1_rep3', 'c2_rep5'], [100, 100, 100], 0.1, {'value': 100}),
        ('peptides', 'pept424', "fixed", ['c1_rep2', 'c1_rep3', 'c2_rep5'], [100, 100, 100], 1.5, {'value': 100, 'random_noise': True}),
        ('peptides', 'pept424', "global min", ['c1_rep2', 'c1_rep3', 'c2_rep5'], [1.91, 1.91, 1.91], 0.1, {}),
        ('peptides', 'pept424', "global min", ['c1_rep2', 'c1_rep3', 'c2_rep5'], [0.914, 0.914, 0.914], 0.1, {'shift': 1}),
        ('peptides', 'pept424', "global min", ['c1_rep2', 'c1_rep3', 'c2_rep5'], [0.914, 0.914, 0.914], 2.5, {'shift': 1, 'random_noise': True}),
    ]
)
def test_missing_value_imputation_with_group_mean(
        proteins_dataset: ProteinsDataset,
        peptides_dataset: PeptidesDataset,
        dset_name,
        row_idx,
        impute_method,
        targ_cols,
        targ_values,
        a_tol,
        kwargs):
    # setup
    dsets = {
        'proteins': proteins_dataset,
        'peptides': peptides_dataset
    }
    raw_dataset = dsets[dset_name]

    transformed_dataset = raw_dataset.log2_transform()

    # action
    imputed_dataset = transformed_dataset.impute(method=impute_method, **kwargs)
    data = imputed_dataset.to_table()
    targ_row = data.loc[data.index == row_idx].copy()

    targ_row[targ_cols]

    # assertion
    for i, targ_col in enumerate(targ_cols):
        assert bool(np.isclose(targ_row[targ_col].values[0], targ_values[i], atol=a_tol))
