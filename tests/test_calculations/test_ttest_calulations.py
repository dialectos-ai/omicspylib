import pytest

from omicspylib import ProteinsDataset
from omicspylib.calculations.ttest import calc_ttest_adj


@pytest.mark.parametrize(
    "padj_method, exp_cols, skipped_cols",
    [
        ("fdr_bh", ["p-value", "t-statistic", "adj-p-value"], []),
        (None, ["p-value", "t-statistic"], ["adj-p-value"]),
    ],
)
def test_ttest_calculations(
    proteins_dataset: ProteinsDataset, padj_method, exp_cols, skipped_cols
) -> None:
    # setup
    dataset = proteins_dataset.filter(cond=["c1", "c2"])

    # action
    ttest_out = calc_ttest_adj(
        dataset, condition_a="c1", condition_b="c2", pval_adj_method=padj_method
    )

    # assertion
    for col in exp_cols:
        assert col in ttest_out.columns
    for col in skipped_cols:
        assert col not in ttest_out.columns
    assert ttest_out.shape[0] == dataset.n_records


@pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")
def test_adjusted_pvalues_ignore_undefined_pvalues() -> None:
    """
    A record with an undefined p-value (here a constant row) must not turn
    the adjusted p-values of all other records into NaN.
    """
    import numpy as np
    import pandas as pd
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(0)
    cols = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]
    data = pd.DataFrame(rng.uniform(100, 1000, (5, 8)), columns=cols)
    data.iloc[0, :] = 500.0  # constant row -> undefined t-test
    data.insert(0, "pid", [f"p{i}" for i in range(5)])
    dataset = ProteinsDataset.from_df(data, id_col="pid", conditions={"A": cols[:4], "B": cols[4:]})

    result = calc_ttest_adj(dataset, condition_a="A", condition_b="B", pval_adj_method="fdr_bh")

    assert np.isnan(result.loc["p0", "p-value"])
    assert np.isnan(result.loc["p0", "adj-p-value"])
    # the remaining records are adjusted among themselves only
    valid = result.drop(index="p0")
    expected = multipletests(valid["p-value"], method="fdr_bh")[1]
    np.testing.assert_allclose(valid["adj-p-value"], expected)
