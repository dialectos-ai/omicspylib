"""
Missing value handling shared by all dataset objects.
"""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd

ArrayLike = TypeVar("ArrayLike", pd.DataFrame, pd.Series, np.ndarray)


class MissingValuesMixin:
    """
    The library's definition of a missing value, shared by the dataset and
    experimental condition objects.

    Missing values are stored as ``0`` or ``NaN``. A value is considered
    missing if it is below or equal to a threshold (``na_threshold``,
    ``0.0`` by default). Dataset methods use these helpers instead of
    comparing against the threshold inline, so the definition stays in
    one place.
    """

    @staticmethod
    def _detection_mask(data: ArrayLike, na_threshold: float = 0.0) -> ArrayLike:
        """
        Boolean mask of the detected (non-missing) values.

        A value is detected if it is strictly greater than ``na_threshold``.
        ``NaN`` is never detected.

        Parameters
        ----------
        data : pd.DataFrame, pd.Series or np.ndarray
            Quantitative values.
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.
            Default is ``0.0``.

        Returns
        -------
        pd.DataFrame, pd.Series or np.ndarray
            Boolean mask of the same type and shape as ``data``. ``True``
            where the value is detected.
        """
        return data > na_threshold

    @classmethod
    def _mask_missing(cls, data: pd.DataFrame, na_threshold: float = 0.0) -> pd.DataFrame:
        """
        Replace missing values with ``NaN``.

        Parameters
        ----------
        data : pd.DataFrame
            Quantitative values.
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.
            Default is ``0.0``.

        Returns
        -------
        pd.DataFrame
            A copy of ``data``, with missing values replaced by ``NaN``. The
            input is not modified.
        """
        return data.where(cls._detection_mask(data, na_threshold))
