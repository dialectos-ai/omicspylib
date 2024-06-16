from omicspylib import PairwiseComparisonTTestFC


def test_pairwise_ttest_fold_change_comparison_success_case(proteins_dataset):
    # setup
    exp_cols = [
        'p-value', 't-statistic', 'adj-p-value',
        'fold change', 'log2 fold change'
    ]
    comparison = PairwiseComparisonTTestFC(proteins_dataset, condition_a='c1', condition_b='c2')

    # action
    result = comparison.compare(pval_adj_method='fdr_bh')

    # assertion
    for col in exp_cols:
        assert col in result.columns
