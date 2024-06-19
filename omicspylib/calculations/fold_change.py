import numpy as np
import pandas as pd


def calc_fold_change(data: pd.DataFrame, condition_a: str, condition_b: str) -> pd.DataFrame:
    """
    Calculates fold change of condition a over condition b.
    Log2 values of fold change are also calculated, so that
    you can use them for plotting. Note the 2x fold increase
    has log2 = 1 and 2x fold decrease has log2 value = -1.

    Use identifiers as index in the provided data.

    Parameters
    ----------
    data: pd.DataFrame
        A table with averaged values.
    condition_a: str
        Column name of condition a.
    condition_b: str
        Column name of condition b.

    Returns
    -------
    pd.DataFrame
        A data frame with fold change of condition a over condition b.
    """
    fc = data[condition_a] / data[condition_b]
    log2_fc = np.log2(fc)

    return pd.DataFrame({
        'fold change': fc,
        'log2 fold change': log2_fc
    })
