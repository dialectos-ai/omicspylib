from omicspylib import PairwiseComparisonTTestFC, PairwiseUniqueEntryComparison


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


def test_unique_entry_comparison_success_case(proteins_dataset):
    # setup
    unique_selector = PairwiseUniqueEntryComparison(
        proteins_dataset, condition_a='c1', condition_b='c2')

    # action
    results = unique_selector.compare(
        majority_grp_min_freq=3,
        minority_grp_max_freq=0,
        na_threshold=0.0)

    # assertion
    frequencies = results.reset_index()\
        .groupby('frequency_class')\
        .count()\
        .to_dict(orient='index')
    assert frequencies['common']['protein_id'] == 89
    assert frequencies['none']['protein_id'] == 2
    assert frequencies['unique c1']['protein_id'] == 3
    assert frequencies['unique c2']['protein_id'] == 6
