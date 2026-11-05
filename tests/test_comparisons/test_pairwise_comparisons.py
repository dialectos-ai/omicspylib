from omicspylib import PairwiseComparisonTTestFC, PairwiseUniqueEntryComparison
from omicspylib.datasets.proteins import ProteinsDataset


def test_pairwise_ttest_fold_change_comparison_success_case(
    proteins_dataset: ProteinsDataset,
) -> None:
    # setup
    exp_cols = [
        "p-value",
        "t-statistic",
        "adj-p-value",
        "fold change",
        "log2 fold change",
    ]
    comparison = PairwiseComparisonTTestFC(
        proteins_dataset, condition_a="c1", condition_b="c2"
    )

    # action
    result = comparison.eval(pval_adj_method="fdr_bh")

    # assertion
    for col in exp_cols:
        assert col in result.columns


def test_unique_entry_comparison_success_case(
    proteins_dataset: ProteinsDataset,
) -> None:
    # setup
    unique_selector = PairwiseUniqueEntryComparison(
        proteins_dataset, condition_a="c1", condition_b="c2"
    )

    # action
    results = unique_selector.eval(
        majority_grp_min_freq=3, minority_grp_max_freq=0, na_threshold=0.0
    )

    # assertion
    frequencies = (
        results.reset_index().groupby("frequency_class").count().to_dict(orient="index")
    )
    assert frequencies["common"]["protein_id"] == 89
    assert frequencies["none"]["protein_id"] == 2
    assert frequencies["unique c1"]["protein_id"] == 3
    assert frequencies["unique c2"]["protein_id"] == 6


def _unique_entries_dataset() -> ProteinsDataset:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    cols = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)]
    data = pd.DataFrame(rng.uniform(100, 1000, (4, 10)), columns=cols)
    data.iloc[0, 5:] = 0  # p0: in all of A, none of B -> unique A
    data.iloc[1, :5] = 0  # p1: in all of B, none of A -> unique B
    data.iloc[2, :] = 0  # p2: missing everywhere -> none
    data.insert(0, "pid", [f"p{i}" for i in range(4)])  # p3: common
    return ProteinsDataset.from_df(data, id_col="pid", conditions={"A": cols[:5], "B": cols[5:]})


def test_unique_entry_comparison_classes() -> None:
    comparison = PairwiseUniqueEntryComparison(_unique_entries_dataset(), condition_a="A", condition_b="B")

    result = comparison.eval(majority_grp_min_freq=4, minority_grp_max_freq=0)

    assert result["frequency_class"].to_dict() == {
        "p0": "unique A", "p1": "unique B", "p2": "none", "p3": "common"
    }


def test_unique_entry_comparison_after_filtering() -> None:
    """
    Filtering removes records from the conditions they are missing in.
    These records must still be counted as missing, and keep their class.
    """
    dataset = _unique_entries_dataset().filter(min_frequency=3)
    comparison = PairwiseUniqueEntryComparison(dataset, condition_a="A", condition_b="B")

    result = comparison.eval(majority_grp_min_freq=4, minority_grp_max_freq=0)

    assert result["frequency_class"].to_dict() == {
        "p0": "unique A", "p1": "unique B", "p3": "common"
    }
