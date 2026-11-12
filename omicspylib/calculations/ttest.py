from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from omicspylib.datasets.proteins import ProteinsDataset

MULTITEST_METHOD = Literal[
    "bonferroni",
    "sidak",
    "holm-sidak",
    "holm",
    "simes-hochberg",
    "hommel",
    "fdr_bh",
    "fdr_by",
    "fdr_tsbh",
    "fdr_tsbky",
]


def calc_ttest_adj(
        data: ProteinsDataset,
        condition_a: str,
        condition_b: str,
        na_threshold: float = 0.0,
        pval_adj_method: MULTITEST_METHOD | None = "fdr_bh") -> pd.DataFrame:
    """
    Calculate t-test and correct p-values for multiple-hypothesis testing error.

    Parameters
    ----------
    data: ProteinsDataset
        A proteins dataset object.
    condition_a: str
        Name of condition A to be evaluated.
    condition_b: str
        Name of condition B to be evaluated.
    na_threshold: float
        Threshold for NaN values.
    pval_adj_method: MULTITEST_METHOD, optional
        Method to adjust p-values for multiple-hypothesis testing.
        By default, Benjamini/Hochberg  (non-negative) (`fdr_bh`)
        is selected.

    Returns
    -------
    pd.DataFrame
        A pandas data frame with the calculated p-values, t-statistic
        and optionally adjusted p-values. Row indices remain as they
        were provided.
    """
    df = data.to_table(join_method="inner")
    mask = df > na_threshold
    df[~mask] = np.nan
    a_cols = data.experiment_names(condition_a)
    b_cols = data.experiment_names(condition_b)
    m1 = df[a_cols].to_numpy()
    m2 = df[b_cols].to_numpy()

    # I assume that you removed cases with low frequency before
    t_stat, p_val = ttest_ind(m1, m2, axis=1, nan_policy="omit")

    p_val = np.asarray(p_val, dtype=float)
    out_data = {"p-value": p_val, "t-statistic": t_stat}
    if pval_adj_method is not None:
        # The p-value is undefined (NaN) for records that cannot be tested,
        # e.g. with constant values or fewer than two values in a group.
        # Adjust the valid p-values among themselves; a single NaN passed to
        # ``multipletests`` would turn every adjusted p-value into NaN.
        adjusted_p_values = np.full_like(p_val, np.nan)
        is_valid = ~np.isnan(p_val)
        if is_valid.any():
            _, adjusted_p_values[is_valid], _, _ = multipletests(
                p_val[is_valid], method=pval_adj_method, is_sorted=False)
        out_data["adj-p-value"] = adjusted_p_values

    return pd.DataFrame(out_data, index=df.index)
