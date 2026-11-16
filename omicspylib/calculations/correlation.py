"""
Pairwise correlation between experiments.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

CorrelationMethod = Literal["spearman", "pearson", "kendall"]

CORRELATION_COLUMNS = ["experiment_a", "experiment_b", "coefficient", "n_records"]


def calc_pairwise_correlation(
    data: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    method: CorrelationMethod = "spearman",
    min_records: int = 10,
) -> pd.DataFrame:
    """
    Correlation coefficient for each pair of columns, on pairwise-complete records.

    For every pair, only the records with a value in both columns are used.
    Missing values must be ``NaN``. A ``0`` is treated as a measured value,
    so mask missing values before calling this function (e.g., with
    ``to_table(mask_missing=True)`` on a dataset).

    No transformation is applied. Pearson's coefficient is sensitive to the
    scale of the values, so log-transform intensities before using it.

    Parameters
    ----------
    data : pd.DataFrame
        One column per experiment, one row per record. ``NaN`` marks a
        missing value.
    pairs : Sequence[tuple[str, str]]
        Pairs of column names to correlate.
    method : {"spearman", "pearson", "kendall"}, optional
        Correlation coefficient. Default is ``"spearman"``.
    min_records : int, optional
        Minimum number of records with a value in both columns. Pairs with
        fewer records get a ``NaN`` coefficient. Default is ``10``.

    Returns
    -------
    pd.DataFrame
        One row per pair, with columns ``experiment_a``, ``experiment_b``,
        ``coefficient`` and ``n_records`` (the number of records with a
        value in both experiments).
    """
    if method not in ("spearman", "pearson", "kendall"):
        raise ValueError(
            f"method must be 'spearman', 'pearson' or 'kendall', got {method!r}."
        )
    if min_records < 2:
        raise ValueError(f"min_records must be at least 2, got {min_records}.")

    if len(pairs) == 0:
        return pd.DataFrame(columns=CORRELATION_COLUMNS).astype(
            {"coefficient": float, "n_records": int}
        )

    columns = list(dict.fromkeys(name for pair in pairs for name in pair))
    missing_cols = [c for c in columns if c not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in data: {missing_cols}.")

    values = data[columns].astype(float)
    # DataFrame.corr uses pairwise-complete observations, ranking them per
    # pair for spearman and kendall.
    coefficients = values.corr(method=method, min_periods=min_records).to_numpy(dtype=float)
    detected = values.notna().to_numpy(dtype=np.int64)
    counts = detected.T @ detected
    position = {name: i for i, name in enumerate(columns)}
    idx_a = [position[a] for a, _ in pairs]
    idx_b = [position[b] for _, b in pairs]

    return pd.DataFrame(
        {
            "experiment_a": [a for a, _ in pairs],
            "experiment_b": [b for _, b in pairs],
            "coefficient": coefficients[idx_a, idx_b],
            "n_records": counts[idx_a, idx_b],
        }
    )
