from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from omicspylib import ProteinsDataset

MULTITEST_METHOD = Literal[
    'bonferroni',
    'sidak',
    'holm-sidak',
    'holm',
    'simes-hochberg',
    'hommel',
    'fdr_bh',
    'fdr_by',
    'fdr_tsbh',
    'fdr_tsbky',
]


def calc_ttest_adj(
        data: ProteinsDataset,
        condition_a: str,
        condition_b: str,
        na_threshold: float = 0.0,
        pval_adj_method: Optional[MULTITEST_METHOD] = 'fdr_bh') -> pd.DataFrame:
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
    df = data.to_table(join_method='inner')
    mask = df > na_threshold
    df[~mask] = np.nan
    a_cols = data.experiments(condition_a)
    b_cols = data.experiments(condition_b)
    m1 = df[a_cols].values
    m2 = df[b_cols].values

    # I assume that you removed cases with low frequency before
    t_stat, p_val = ttest_ind(m1, m2, axis=1, nan_policy='omit')

    out_data = {'p-value': p_val, 't-statistic': t_stat}
    if pval_adj_method is not None:
        reject, adjusted_p_values, _, _ = multipletests(
            p_val, method=pval_adj_method, is_sorted=False)
        out_data['adj-p-value'] = adjusted_p_values

    return pd.DataFrame(out_data, index=df.index)
