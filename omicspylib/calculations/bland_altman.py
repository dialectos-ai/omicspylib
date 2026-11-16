"""
Bland-Altman agreement between experiments.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

BLAND_ALTMAN_COLUMNS = [
    "experiment_a", "experiment_b", "bias", "sd", "loa_lower", "loa_upper", "n_records",
]

# Limits of agreement contain ~95% of the differences, assuming they are
# normally distributed.
LOA_Z = 1.96


def calc_bland_altman(
    data: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    min_records: int = 10,
) -> pd.DataFrame:
    """
    Bias and limits of agreement for each pair of columns.

    For each pair, the differences ``a - b`` are calculated on the records
    with a value in both columns (``NaN`` marks a missing value). The bias
    is their mean and the limits of agreement are ``bias ± 1.96 * sd``,
    with ``sd`` their sample standard deviation.

    No transformation is applied. For intensities, use log-transformed
    values, so the differences are log fold changes.

    Parameters
    ----------
    data : pd.DataFrame
        One column per experiment, one row per record. ``NaN`` marks a
        missing value.
    pairs : Sequence[tuple[str, str]]
        Pairs of column names to compare.
    min_records : int, optional
        Minimum number of records with a value in both columns. Pairs with
        fewer records get ``NaN`` values. Default is ``10``.

    Returns
    -------
    pd.DataFrame
        One row per pair, with columns ``experiment_a``, ``experiment_b``,
        ``bias``, ``sd``, ``loa_lower``, ``loa_upper`` and ``n_records``.
    """
    if min_records < 2:
        raise ValueError(f"min_records must be at least 2, got {min_records}.")

    rows = []
    for exp_a, exp_b in pairs:
        diff = (data[exp_a].astype(float) - data[exp_b].astype(float)).dropna()
        n_records = len(diff)
        if n_records >= min_records:
            bias = float(diff.mean())
            sd = float(diff.std(ddof=1))
        else:
            bias = sd = float("nan")
        rows.append({
            "experiment_a": exp_a,
            "experiment_b": exp_b,
            "bias": bias,
            "sd": sd,
            "loa_lower": bias - LOA_Z * sd,
            "loa_upper": bias + LOA_Z * sd,
            "n_records": n_records,
        })

    result = pd.DataFrame(rows, columns=BLAND_ALTMAN_COLUMNS)
    return result.astype({"bias": float, "sd": float, "loa_lower": float,
                          "loa_upper": float, "n_records": np.int64})
