import numpy as np
import pandas as pd


def calc_fold_change(data: pd.DataFrame, condition_a: str, condition_b: str) -> pd.DataFrame:
    fc = data[condition_a] / data[condition_b]
    log2_fc = np.log2(fc)
    return pd.DataFrame({
        f'fold change {condition_a} over {condition_b}': fc,
        f'log2 fold change {condition_a} over {condition_b}': log2_fc
    })
