import pytest

from omicspylib import ProteinsDataset
from omicspylib.calculations.ttest import calc_ttest_adj


@pytest.mark.parametrize(
    "padj_method, exp_cols, skipped_cols", [
        ('fdr_bh', ['p-value', 't-statistic', 'adj-p-value'], []),
        (None, ['p-value', 't-statistic'], ['adj-p-value']),
    ]
)
def test_ttest_calculations(proteins_dataset: ProteinsDataset, padj_method, exp_cols, skipped_cols):
    # setup
    dataset = proteins_dataset.filter(cond=['c1', 'c2'])

    # action
    ttest_out = calc_ttest_adj(dataset, condition_a='c1', condition_b='c2', pval_adj_method=padj_method)

    # assertion
    for col in exp_cols:
        assert col in ttest_out.columns
    for col in skipped_cols:
        assert col not in ttest_out.columns
    assert ttest_out.shape[0] == dataset.n_records
