import numpy as np
import pytest

from omicspylib import ProteinsDataset


@pytest.mark.parametrize(
    "impute_method,targ_cols,targ_values,a_tol,kwargs",
    [
        ("fixed", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [100, 100, 100], 0.1, {'value': 100}),
        ("global min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.046, 20.046, 20.046], 0.1, {}),
        ("global min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.046, 15.046, 15.046], 0.1, {'shift': 5}),
        ("global mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.82, 20.82, 20.82], 0.1, {}),
        ("global median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.83, 20.83, 20.83], 0.1, {}),
        ("global row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.22, 20.22, 20.22], 0.1, {}),
        ("global row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.22, 15.22, 15.22], 0.1, {'shift': 5}),
        ("global row mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.77, 20.77, 20.77], 0.1, {}),
        ("global row median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.74, 20.74, 20.74], 0.1, {}),
        ("group row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [20.95, 20.95, 20.22], 0.1, {}),
        ("group row min", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [15.95, 15.95, 15.22], 0.1, {'shift': 5}),
        ("group row mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [21.03, 21.03, 20.52], 0.1, {}),
        ("group row mean", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [16.03, 16.03, 15.52], 0.1, {'shift': 5}),
        ("group row median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [21.03, 21.03, 20.56], 0.1, {}),
        ("group row median", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [16.03, 16.03, 15.56], 0.1, {'shift': 5}),
        ("group row mean with noise", ['c1_rep1', 'c1_rep2', 'c2_rep4'], [21.03, 21.03, 20.52], 0.7, {}),
    ]
)
def test_missing_value_imputation_with_group_mean(
        proteins_dataset: ProteinsDataset,
        impute_method,
        targ_cols,
        targ_values,
        a_tol,
        kwargs):
    # setup
    transformed_dataset = proteins_dataset.log2_transform()

    # action
    dataset = transformed_dataset.impute(method=impute_method, **kwargs)
    data = dataset.to_table()
    p5_row = data.loc[data.index == 'p5'].copy()

    # assertion
    for i, targ_col in enumerate(targ_cols):
        assert bool(np.isclose(p5_row[targ_col].values[0], targ_values[i], atol=a_tol))
