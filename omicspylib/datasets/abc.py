from __future__ import annotations

import abc
import copy
import warnings
from collections.abc import Sequence
from functools import reduce
from typing import Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd
from typing_extensions import Self

from omicspylib.calculations.bland_altman import calc_bland_altman
from omicspylib.calculations.correlation import (
    CorrelationMethod,
    calc_pairwise_correlation,
)
from omicspylib.datasets.missing import MissingValuesMixin
from omicspylib.utils.deprecation import resolve_renamed_arg

AxisName = Literal["rows", "columns"]
ExperimentPairs = Literal["within_conditions", "across_conditions", "all"]
MissingHandling = Literal["pairwise", "listwise"]
NormMethod = Literal["mean", "median"]  # future implementation for quantile etc
MergeHow = Literal["left", "right", "inner", "outer", "cross"]
ConditionImputeMethod = Literal[
    "fixed", "fixed row", "row min", "row mean", "row median"
]
ImputeMethod = Literal[
    "fixed",
    "global min",
    "global mean",
    "global median",
    "global row min",
    "global row median",
    "global row mean",
    "group row min",
    "group row mean",
    "group row median",
]

T = TypeVar("T", bound="TabularDataset")


def _trimmed_mean(values: pd.Series, trim_fraction: float) -> float:
    """
    Mean after discarding ``trim_fraction / 2`` of the values from each end.

    NaNs are ignored. If trimming would remove every value, or the number
    of values to cut from each end rounds down to zero, the plain mean
    is returned.
    """
    arr = np.sort(values.dropna().to_numpy(dtype=float))
    k = int(len(arr) * (trim_fraction / 2))
    if k > 0 and 2 * k < len(arr):
        arr = arr[k:-k]
    return float(arr.mean()) if len(arr) > 0 else float("nan")


def _central_value(values: pd.Series, method: NormMethod, trim_fraction: float) -> float:
    """
    Central value of ``values`` used as the normalization anchor.

    ``mean`` returns the (optionally trimmed) mean, ``median`` the median.
    NaNs are ignored. ``trim_fraction`` does not apply to the median,
    since symmetric trimming leaves it unchanged.
    """
    if method == "mean":
        return _trimmed_mean(values, trim_fraction)
    if method == "median":
        arr = values.dropna().to_numpy(dtype=float)
        return float(np.median(arr)) if len(arr) > 0 else float("nan")
    raise NotImplementedError(f"Normalization method {method} is not implemented.")


class TabularExperimentalConditionDataset(MissingValuesMixin, abc.ABC):
    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        id_col: str,
        experiment_cols: list,
        **kwargs,
    ) -> None:
        """
        Create an experimental condition from a pandas data frame.

        Parameters
        ----------
        name: str
            Name of the experimental condition (e.g., ``control``).
        data: pd.DataFrame
            A table with one row per record and one column per experiment.
            It may contain other columns too: only ``id_col`` and
            ``experiment_cols`` are used, and the data frame is copied.
        id_col: str
            Name of the column with the record ids. Ids should be unique.
        experiment_cols: list
            Names of the columns with the experiments of this condition.
        **kwargs
            Ignored. Accepted so that subclasses can share the same signature.
        """
        self._name = name
        self._data: pd.DataFrame = (
            data[[id_col, *experiment_cols]].copy().set_index(id_col)
        )
        self._id_col = id_col
        self._metadata = {}

    @property
    def id_col(self) -> str:
        """
        Column identifier for the record ids.

        Returns
        -------
        str:
            Column name with the unique identifiers as string.
        """
        return self._id_col

    @property
    def _experiments(self) -> list:
        return self._data.columns.tolist()

    @property
    def metadata(self) -> dict:
        """
        Return the dataset's metadata. If no values
        exist, an empty dictionary is returned.

        Returns
        -------
        dict
            Datasets metadata.
        """
        return self._metadata

    @property
    def n_experiments(self) -> int:
        """
        Number of experiments in this condition.

        Returns
        -------
        int
            The number of experiments.
        """
        return len(self._data.columns)

    @property
    def experiment_names(self) -> list[str]:
        """
        Get the list of experiment names.

        Returns
        -------
        list
            A list of the experiment names from the
            given experimental condition.
        """
        return self._experiments

    @property
    def record_ids(self) -> list[str]:
        """
        Ids of the records in this condition, e.g. protein or peptide ids.

        Returns
        -------
        list
            A list of unique record ids.
        """
        return self._data.index.tolist()

    @property
    def name(self) -> str:
        """
        Get experimental condition name (e.g. treated, untreated etc.).

        Returns
        -------
        str
            Dataset's name.
        """
        return self._name

    def describe(self, na_threshold: float = 0.0) -> dict:
        """
        Returns basic information about the dataset.

        Parameters
        ----------
        na_threshold: float, optional
            Values below or equal to this threshold are considered missing,
            and are not counted in ``n_records_per_experiment``.

        Returns
        -------
        dict
            Dataset's basic information including name, number of
            experiments, number of records, number of experimental names
            and number of records per experiment.
        """
        return {
            "name": self._name,
            "n_experiments": self.n_experiments,
            "n_records": len(self.record_ids),
            "experiment_names": self._data.columns.tolist(),
            "n_records_per_experiment": self._detection_mask(self._data, na_threshold).sum(axis=0).tolist(),
        }

    def min(
        self, na_threshold: float = 0.0, axis: AxisName | None = None
    ) -> float | pd.Series:
        """
        Calculate the minimum value, ignoring missing values.

        Missing values (by default, values ``<= 0.0``) are left out, so that
        the minimum is the lowest value actually measured.

        Parameters
        ----------
        na_threshold: float
            Values below or equal to this threshold are considered missing.
        axis: AxisName, optional
            * ``None`` (default): a single minimum across all values.
            * ``"rows"``: the minimum of each experiment, calculated down
              the rows.
            * ``"columns"``: the minimum of each record, calculated across
              the experiments.

        Returns
        -------
        float or pd.Series
            A single value if ``axis`` is ``None``. Otherwise, a series indexed
            by experiment name (``"rows"``) or by record id (``"columns"``).
        """
        df = self._mask_missing(self._data, na_threshold)
        if axis is None:
            min_value = np.nanmin(df.to_numpy().flatten())
        elif axis == "rows":
            min_value = df.min(axis=0)
        else:  # axis = columns
            min_value = df.min(axis=1)

        return min_value

    def missing_values(
        self, na_threshold: float = 0.0
    ) -> tuple[pd.DataFrame, int, int]:
        """
        Calculate the number of missing values per experiment.

        Parameters
        ----------
        na_threshold : float, optional
            Values equal or below this threshold will be considered missing.

        Returns
        -------
        pd.DataFrame
            A Pandas data frame with the number of missing values per experiment.
        Int
            Number of missing values in total.
        Int
            Number of total values of that condition.
        """
        n_missing_per_exp = self._data.shape[0] - np.sum(
            self._detection_mask(self._data, na_threshold), axis=0
        )
        n_missing_total = np.sum(n_missing_per_exp)
        total_values = self._data.shape[0] * self._data.shape[1]
        df = pd.DataFrame(
            {
                "experiment": self._data.columns,
                "n_missing": n_missing_per_exp.tolist(),
                "condition": self._name,
            }
        )
        return df, int(n_missing_total), int(total_values)

    def log2_transform(self) -> Self:
        """
        Perform log2 transformation on all experiments of the condition,
        as ``log2(x + 1)``. The ``+ 1`` keeps zeros (missing values) at zero.

        The values are modified in place.

        Returns
        -------
        Self
            The same object, with the values transformed.
        """
        self._data = np.log2(self._data + 1)  # type: ignore
        return self

    def log2_backtransform(self) -> Self:
        """
        Invert :meth:`log2_transform`, as ``2 ** x - 1``, converting
        values back to their original scale.

        The values are modified in place.

        Returns
        -------
        Self
            The same object, with the values transformed.
        """
        self._data = 2**self._data - 1
        return self

    def mean(self, na_threshold: float = 0.0, axis: Literal[0, 1] = 1) -> pd.DataFrame:
        """
        Calculate the mean value, ignoring missing values.

        Parameters
        ----------
        na_threshold : float
            Values below or equal to this threshold are considered missing
            and are excluded from the calculation.
        axis: int
            Axis to perform the operation. Use ``1`` for row by row (mean of each
            record across experiments) and ``0`` for column by column (mean of each
            experiment across records).

        Returns
        -------
        pd.DataFrame
            For ``axis=1``, a data frame indexed by record id with a single
            column named ``mean_<condition name>``. For ``axis=0``, a data frame
            indexed by experiment name with a single column named ``mean``.
        """
        mask = self._detection_mask(self._data, na_threshold)
        data = self._data.where(mask)
        mean = data.sum(axis=axis) / mask.sum(axis=axis)
        if axis == 1:  # row mean
            return pd.DataFrame({f"mean_{self.name}": mean})
        # column mean
        return pd.DataFrame({"mean": mean})

    @abc.abstractmethod
    def filter(
        self,
        experiments: str | list | None = None,
        min_frequency: int | None = None,
        na_threshold: float = 0.0,
        ids: list | None = None,
        *,
        exp: str | list | None = None,
    ) -> Self:
        raise NotImplementedError

    def _apply_filter(self, experiments, min_frequency, na_threshold, ids=None) -> pd.DataFrame:
        data = self._data.copy()
        if min_frequency is not None:
            valid_rows = self._detection_mask(data, na_threshold).sum(axis=1) >= min_frequency
            data = data.loc[valid_rows, :].copy()
        if isinstance(experiments, str):
            experiments = [experiments]
        if experiments is not None:
            # You might filter by providing names across experimental
            # conditions. So when you work on each condition separately,
            # some names are no longer valid.
            local_exp = [ex for ex in experiments if ex in data.columns]
            data = data.loc[:, local_exp].copy()
        if ids is not None:
            data = data.loc[data.index.isin(ids)].copy()

        return cast(pd.DataFrame, data)

    def frequency(self, na_threshold: float = 0.0, axis: int = 1) -> pd.DataFrame:
        """
        Count the number of values above the specified threshold.

        Parameters
        ----------
        na_threshold : float
            Values below or equal to this threshold are considered missing.
        axis: int
            Axis on which to calculate the frequency. Use ``1`` for row by row
            (in how many experiments each record is present) and ``0`` for column
            by column (how many records are present in each experiment).

        Returns
        -------
        pd.DataFrame
            For ``axis=1``, a data frame indexed by record id with a single
            column named ``frequency_<condition name>``. For ``axis=0``, a data
            frame indexed by experiment name with a single column named
            ``frequency``.
        """
        f = np.sum(self._detection_mask(self._data, na_threshold), axis=axis)
        if axis == 1:
            return pd.DataFrame({f"frequency_{self.name}": f})
        return pd.DataFrame({"frequency": f})

    @abc.abstractmethod
    def drop(
        self,
        experiments: str | list | None = None,
        ids: list | None = None,
        omit_missing_cols: bool = True,
        **kwargs,
    ) -> Self:
        raise NotImplementedError

    def _apply_drop(
        self,
        experiments: str | list | None = None,
        ids: list | None = None,
        omit_missing_cols: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        data = self._data.copy()
        if isinstance(experiments, str):
            experiments = [experiments]

        if omit_missing_cols and experiments is not None:
            # allow the user to pass column names that don't exist or
            # are already excluded from previous steps.
            experiments = [e for e in experiments if e in data.columns]

        if experiments is not None:
            data = data.drop(experiments, axis=1)

        if ids is not None:
            data = data.loc[~data.index.isin(ids)].copy()

        return data

    def _calc_mean_std(self) -> float:
        """
        Calculate the average standard deviation between repeats
        so that you can use it to add random noise during missing value imputation.
        """
        return self._data.std(axis=1, skipna=True).dropna().mean()

    def impute(
        self,
        method: ConditionImputeMethod,
        na_threshold: float = 0.0,
        value: float | pd.Series | None = None,
        shift: float = 0.0,
        random_noise: bool = False,
    ) -> Self:
        """
        Impute missing values of this condition.

        The values are modified in place.

        Parameters
        ----------
        method: str
            One of:

            * ``fixed``: replace missing values with ``value``, a number.
            * ``fixed row``: replace missing values with a different value
              per record. ``value`` must be a ``pd.Series`` indexed by
              record id.
            * ``row min``, ``row mean``, ``row median``: replace missing
              values of each record with the min, mean or median of its
              values in this condition, minus ``shift``.
        na_threshold: float
            Values below or equal to this threshold are considered missing.
        value: float or pd.Series, optional
            Required by the ``fixed`` and ``fixed row`` methods.
        shift: float
            Subtracted from the imputed value, for the ``row`` methods.
        random_noise: bool
            If ``True``, each imputed value is drawn from a normal
            distribution centered on the selected value, with standard
            deviation equal to the average standard deviation of the
            records in this condition.

        Returns
        -------
        Self
            The same object, with missing values imputed.

        Raises
        ------
        ValueError
            If ``method`` is unknown, or ``value`` is missing for the
            ``fixed`` and ``fixed row`` methods.
        """
        self._data = self._mask_missing(self._data, na_threshold)

        rand_noise_std = self._calc_mean_std() if random_noise else None

        if method == "fixed":
            if value is None:
                raise ValueError(
                    f"To impute missing values with a fixed value,"
                    f" you also need to specify the target fixed value,"
                    f" using the ``value`` argument. Received ``{value}``."
                )
            self._data = self._data.apply(
                lambda row: self._fillna(row, value, std_value=rand_noise_std), axis=1
            )
        elif method == "fixed row":
            if value is None:
                raise ValueError(
                    f"To impute missing values with a fixed row value,"
                    f" you also need to specify the target fixed value array,"
                    f" using the ``value`` argument. Received ``{value}``."
                )
            self._data = self._data.apply(
                lambda row: self._fillna(row, value, std_value=rand_noise_std), axis=1
            )
        elif method == "row min":
            impute_values = self._data.min(axis=1) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values, std_value=rand_noise_std),
                axis=1,
            )
        elif method == "row median":
            impute_values = self._data.median(axis=1, skipna=True) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values, std_value=rand_noise_std),
                axis=1,
            )
        elif method == "row mean":
            impute_values = self._data.mean(axis=1, skipna=True) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values, std_value=rand_noise_std),
                axis=1,
            )
        else:
            raise ValueError(f"Method {method} not implemented")

        return self

    @staticmethod
    def _fillna(
        row: pd.Series, val: pd.Series | float, std_value: float | None = None
    ) -> pd.Series:
        """
        Fill nan values of a Pandas data frame row by row.

        You can either use a fixed value per row, by providing a
        ``pd.Series`` or the same value for all rows, by providing
         a ``float``.  For the first case, the index of the target
        value in the values' array, matches the row.name attribute
        of the row.

        If ``std_value`` is specified, Imputed values will be
        selected from a normal distribution with the mean
        value of the dataset and std the specified value.
        """
        # case where you fill with different value per row
        if isinstance(val, pd.Series):
            val_idx = np.where(row.name == val.index)[0][0]
            if std_value is None:
                row = row.fillna(val.iloc[val_idx])
            else:
                for j, v in enumerate(row):
                    if np.isnan(v):
                        row.iloc[j] = np.random.normal(val.iloc[val_idx], std_value)  # noqa: NPY002
        # case where you fill with the same value across rows
        else:
            if std_value is None:
                row = row.fillna(val)
            else:
                for j, v in enumerate(row):
                    if np.isnan(v):
                        row.iloc[j] = np.random.normal(val, std_value)  # noqa: NPY002

        return row

    def to_table(
        self, mask_missing: bool = False, na_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Return the values of this condition as a pandas data frame.

        Parameters
        ----------
        mask_missing : bool, optional
            If ``True``, missing values (values below or equal to
            ``na_threshold``) are returned as ``NaN``. By default, values
            are returned as stored.
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.
            Used only if ``mask_missing`` is ``True``.

        Returns
        -------
        pd.DataFrame
            A copy of the table, with record ids as the index and one column
            per experiment. Modifying it does not change this object.
        """
        if mask_missing:
            return self._mask_missing(self._data, na_threshold)
        return self._data.copy()

    def detection_matrix(self, na_threshold: float = 0.0) -> pd.DataFrame:
        """
        Return whether each record was detected in each experiment.

        Parameters
        ----------
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.

        Returns
        -------
        pd.DataFrame
            A boolean data frame with the same shape as :meth:`to_table`.
            ``True`` if the value is above ``na_threshold``, ``False`` if it
            is missing.
        """
        return self._detection_mask(self._data, na_threshold)

    def shift(self, exp, value, na_threshold: float = 0.0) -> None:
        """
        Subtract a fixed value from all the values of one experiment.

        Missing values are not changed. The values are modified in place.
        Used by the dataset's ``normalize`` method.

        Parameters
        ----------
        exp: str
            Name of the experiment to shift.
        value: float
            The value to subtract.
        na_threshold : float
            Values below or equal to this threshold are considered missing,
            and are not shifted.
        """
        mask = self._detection_mask(self._data[exp], na_threshold)
        self._data.loc[mask, exp] -= value


class TabularDataset(MissingValuesMixin, abc.ABC):
    def __init__(
        self, conditions: Sequence[TabularExperimentalConditionDataset]
    ) -> None:
        """
        Create a dataset from experimental condition objects.

        You normally don't need to call the constructor directly. Use the
        ``from_df`` class method instead.

        Parameters
        ----------
        conditions: Sequence
            One experimental condition object per condition. Each condition
            name should appear only once.
        """
        self._conditions = conditions

    @classmethod
    @abc.abstractmethod
    def from_df(
        cls, data: pd.DataFrame, id_col: str, conditions: dict[str, list]
    ) -> Self:
        raise NotImplementedError

    @property
    def n_conditions(self) -> int:
        """
        Return the number of experimental conditions included in the dataset.

        Returns
        -------
        int
            Number of experimental conditions included in the dataset.
        """
        return len(self._conditions)

    @property
    def condition_names(self) -> list[str]:
        """
        List experimental condition names.

        Returns
        -------
        list
            A list of experimental condition names.
        """
        return [c.name for c in self._conditions]

    def _append_conditions(
        self,
        new_obj: TabularDataset,
        skip_duplicates: bool = False,
        inplace: bool | None = None,
    ) -> Self:
        """
        Shared implementation of ``append``. Adds the conditions of
        ``new_obj`` after the conditions of ``self``.

        With ``inplace=False``, returns a new dataset and neither input is
        modified. With ``inplace=True``, ``self`` is modified and returned.
        ``inplace=None`` behaves like ``True`` and emits a ``FutureWarning``,
        since the default will change to ``False``.
        In all cases, the result does not share condition objects with ``new_obj``.
        """
        if inplace is None:
            warnings.warn(
                "The default of `inplace` in `append` will change from True to False "
                "in a future release of omicspylib, and `append` will then return a new "
                "dataset without modifying the original. Pass `inplace=True` to keep "
                "the current behaviour, or `inplace=False` to adopt the new one and "
                "silence this warning.",
                category=FutureWarning,
                stacklevel=3,  # user code -> append -> _append_conditions
            )
            inplace = True

        if self.__class__ != new_obj.__class__:
            raise ValueError(
                f"The provided object should be of type {self.__class__.__name__}. "
                f"Received object of type {new_obj.__class__.__name__} instead."
            )
        id_col = self._conditions[0].id_col
        if id_col != new_obj._conditions[0].id_col:
            raise ValueError(
                f"Cannot join, because there is a mismatch between the "
                f"id_col name between the datasets. All datasets should "
                f"have an id_col == {id_col}."
            )

        # Validate all conditions before changing anything, so that a failing
        # call never leaves ``self`` half-modified when ``inplace=True``.
        to_add = []
        existing_names = set(self.condition_names)
        for new_cond in new_obj._conditions:
            if new_cond.name in existing_names:
                if skip_duplicates:
                    continue
                raise ValueError(
                    f"Experimental condition '{new_cond.name}' already exists in "
                    f"the current dataset. Either remove it or set "
                    f"`skip_duplicates=True`."
                )
            to_add.append(new_cond)
            existing_names.add(new_cond.name)

        if inplace:
            self._conditions = [*self._conditions, *copy.deepcopy(to_add)]
            return self

        return self.__class__(conditions=copy.deepcopy([*self._conditions, *to_add]))

    @property
    def n_experiments(self) -> int:
        """
        Returns the number of experiments included in the dataset,
        across all experimental conditions.

        Returns
        -------
        int
            Number of experiments included in the dataset.
        """
        n_exp = 0
        for condition in self._conditions:
            n_exp += condition.n_experiments
        return n_exp

    def experiment_names(self, condition: str | None = None) -> list[str]:
        """
        Get experiment names from the dataset. If an experimental condition
        name is provided, experiment names will be limited to that case.

        Parameters
        ----------
        condition: str or None
            Name of the experimental condition to retrieve names for,
            or `None` to retrieve all experimental conditions.

        Returns
        -------
        list
            A list of experiment names.
        """
        exp_names = []
        for exp_condition in self._conditions:
            if condition is None or exp_condition.name == condition:
                exp_names.extend(exp_condition.experiment_names)

        return exp_names

    @property
    def n_records(self) -> int:
        """
        Returns the number of unique records in the dataset.

        Returns
        -------
        int
            The total number of unique records.
        """
        unique_records = self._get_unique_records()
        return len(unique_records)

    def unique_records(self) -> list:
        """
        Returns a list of unique entry ids across
        all experimental conditions (e.g. unique
        protein accession numbers etc.).

        Returns
        -------
        list
            A list of unique entry ids.
        """
        return self._get_unique_records()

    def _get_unique_records(self) -> list[str]:
        all_records = []
        for condition in self._conditions:
            all_records.extend(condition.record_ids)
        return sorted(set(all_records))

    def describe(self, na_threshold: float = 0.0) -> dict[str, Any]:
        """
        Returns basic information about the dataset.
        Includes fields like number of experimental conditions,
        number of records in total, total number of experiments
        and statistics for each experimental condition.

        Parameters
        ----------
        na_threshold: float, optional
            Values below or equal to this threshold are considered missing,
            and are not counted in the ``n_records_per_experiment`` of each
            condition.

        Returns
        -------
        dict
            Dataset statistics.
        """
        return {
            "n_conditions_total": self.n_conditions,
            "n_records_total": self.n_records,
            "n_experiments_total": self.n_experiments,
            "statistics_per_condition": [
                c.describe(na_threshold=na_threshold) for c in self._conditions
            ],
        }

    def mean(
        self,
        na_threshold: float = 0.0,
        join_method: MergeHow = "inner",
        axis: Literal[0, 1] = 1,
    ) -> pd.DataFrame:
        """
        Calculate the average value for each record within each
        experimental condition and return a merged data frame for
        all conditions.

        Missing values (and values below or equal the specified
        threshold) are omitted.

        By default, and inner join is performed across all conditions.
        Adjust accordingly if needed.

        Parameters
        ----------
        na_threshold: float
            Values below or equal to this threshold are considered missing.
        join_method: MergeHow, optional
            Method of joining records of each experimental
            condition in the output.
        axis: int
            1 for row by row and 0 for column by column.

        Returns
        -------
        pd.DataFrame
            A Pandas data frame containing the average value for
            each condition.
        """
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 or 1, got {axis!r}.")

        tables = [
            c.mean(na_threshold=na_threshold, axis=axis) for c in self._conditions
        ]
        if axis == 1:  # row mean
            return self._join_list_of_tables(tables, how=join_method)

        return pd.concat(tables).transpose()

    def frequency(
        self,
        na_threshold: float = 0.0,
        join_method: MergeHow = "outer",
        axis: int = 1,
        conditions: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the number of experiments within each experimental condition
        with quantitative value above the specified threshold,
        and return a merged data frame for all conditions.

        By default, and outer join is performed across all conditions.
        Adjust accordingly if needed.

        Parameters
        ----------
        na_threshold : float
            Values below or equal to this threshold are considered missing.
        join_method: MergeHow
            Method of joining records of each experimental condition in the output.
        axis: int
            Axis on which to calculate the frequency. Use ``1`` for row by row and
            ``0`` for column by column.
        conditions: List[str], optional
            If specified, only the specified conditions are considered.

        Returns
        -------
        pd.DataFrame
            A Pandas data frame containing the average value for each condition.
        """
        if conditions:
            tables = [
                c.frequency(na_threshold=na_threshold, axis=axis)
                for c in self._conditions
                if c.name in conditions
            ]
        else:
            tables = [
                c.frequency(na_threshold=na_threshold, axis=axis)
                for c in self._conditions
            ]

        if axis == 1:
            return self._join_list_of_tables(tables, how=join_method)

        return pd.concat(tables).transpose()

    def record_counts(
        self,
        na_threshold: float = 0.0,
        value_type: Literal["missing", "present"] = "present",
    ) -> tuple[pd.DataFrame, int, int]:
        """
        Calculate the number of records per experiment that are missing
        or are above a given value threshold.

        Parameters
        ----------
        na_threshold : float
            Values below or equal to this threshold are considered missing,
            or above that threshold are considered present.
        value_type: str
            Either "missing" or "present" to calculate the number of missing
            values or valid records per experiment.


        Returns
        -------
        pd.DataFrame
            A Pandas data frame with the number of missing cases per
            experiment and condition.
        Int
            Number of missing values.
        Int
            Number of values in total

        """
        data = self.to_table()
        if value_type == "missing":
            n_counts_per_exp = data.shape[0] - np.sum(self._detection_mask(data, na_threshold), axis=0)
        elif value_type == "present":
            n_counts_per_exp = np.sum(self._detection_mask(data, na_threshold), axis=0)
        else:
            raise ValueError("type must be either 'missing' or 'present'")  # pyright: ignore

        n_counts_df = n_counts_per_exp.reset_index()
        n_counts_df.columns = ["experiment", "n_counts"]

        # total number of missing values across all datasets.
        n_counts_total = np.sum(n_counts_per_exp)
        n_total = data.shape[0] * len(data.columns)
        names_lookup = [
            {"experiment": exp, "condition": cond.name}
            for cond in self._conditions
            for exp in cond.experiment_names
        ]
        names_lookup_df = pd.DataFrame(names_lookup)
        out_df = n_counts_df.merge(names_lookup_df, on="experiment", how="left")

        return out_df, n_counts_total, n_total

    def drop(
        self,
        experiments: str | list | None = None,
        conditions: str | list | None = None,
        ids: list | None = None,
        *,
        exp: str | list | None = None,
        cond: str | list | None = None,
    ) -> Self:
        """
        Drop specified experiment(s) and or condition(s).

        Parameters
        ----------
        experiments: str, list, optional
            Experiment name(s) to be dropped.
        conditions: str, list, optional
            Experimental condition(s) to be dropped.
        ids: list, optional
            If specified, a list of records ids to drop from the dataset.
        exp: str, list, optional
            .. deprecated:: 0.3.0
                Use ``experiments`` instead.
        cond: str, list, optional
            .. deprecated:: 0.3.0
                Use ``conditions`` instead.

        Returns
        -------
        Self
            An object of the same instance type without
            the specified experiment(s) and/or condition(s).
        """
        experiments = resolve_renamed_arg(experiments, exp, "experiments", "exp")
        conditions = resolve_renamed_arg(conditions, cond, "conditions", "cond")

        filt_conditions = copy.deepcopy(self._conditions)
        if isinstance(experiments, str):
            experiments = [experiments]
        if isinstance(conditions, str):
            conditions = [conditions]
        if conditions is not None:
            filt_conditions = [c for c in filt_conditions if c.name not in conditions]

        if experiments is not None:
            filt_conditions = [c.drop(experiments) for c in filt_conditions]

        if ids is not None:
            filt_conditions = [c.drop(ids=ids) for c in filt_conditions]

        return self.__class__(conditions=filt_conditions)

    def filter(
        self,
        experiments: str | list | None = None,
        conditions: str | list | None = None,
        min_frequency: int | None = None,
        na_threshold: float = 0.0,
        ids: list | None = None,
        *,
        exp: str | list | None = None,
        cond: str | list | None = None,
    ) -> Self:
        """
        Filter the dataset based on a given set of properties.

        Parameters
        ----------
        experiments: list, str, optional
            List of experiment name or a single experiment to keep.
            Leave empty to keep all experiments.
        conditions: list, str, optional
            List of experimental condition names. If provided, only the conditions
            specified will remain in the dataset.
        min_frequency: int or None, optional
            If specified, records of the dataset will be filtered to the records with
            greater than or equal the specified frequency.
        na_threshold: float or None, optional
            Values below or equal to this threshold are considered missing.
            It is used in to filter records based on the number of missing values.
        ids: list, optional
            If specified, a list of records ids to keep in the dataset.
        exp: list, str, optional
            .. deprecated:: 0.3.0
                Use ``experiments`` instead.
        cond: list, str, optional
            .. deprecated:: 0.3.0
                Use ``conditions`` instead.

        Returns
        -------
        ProteinsDataset
            A new instance of the dataset object, filtered based on the
            user's input.
        """
        experiments = resolve_renamed_arg(experiments, exp, "experiments", "exp")
        conditions = resolve_renamed_arg(conditions, cond, "conditions", "cond")

        exp_conditions = list(self._conditions).copy()
        if isinstance(experiments, str):
            experiments = [experiments]
        if isinstance(conditions, str):
            conditions = [conditions]

        if conditions is not None:
            exp_conditions = [c for c in exp_conditions if c.name in conditions]

        filt_min_f = min_frequency is not None
        filt_na_th = na_threshold is not None
        filt_ids = ids is not None
        if filt_min_f or filt_na_th or filt_ids:
            exp_conditions = [
                c.filter(
                    experiments=experiments,
                    min_frequency=min_frequency,
                    na_threshold=na_threshold,
                    ids=ids,
                )
                for c in exp_conditions
            ]

        return self.__class__(conditions=exp_conditions)

    @staticmethod
    def _join_list_of_tables(
        tables: list[pd.DataFrame], how: MergeHow = "outer"
    ) -> pd.DataFrame:
        return reduce(
            lambda left, right: left.merge(
                right, left_index=True, right_index=True, how=how
            ),
            tables,
        )

    def log2_transform(self) -> Self:
        """
        Perform log2 transformation in all experiments.

        Returns
        -------
        An object of the same instance with the values transformed.
        """
        conditions_copy = copy.deepcopy(self._conditions)
        log2_conditions = [c.log2_transform() for c in conditions_copy]
        return self.__class__(conditions=log2_conditions)

    def log2_backtransform(self) -> Self:
        """
        Calculate the exponential with base 2.
        Is used to invert log2 transformation and convert values
        back to their original scale.

        Returns
        -------
        An object of the same instance with the values transformed.
        """
        conditions_copy = copy.deepcopy(self._conditions)
        bt_conditions = [c.log2_backtransform() for c in conditions_copy]
        return self.__class__(conditions=bt_conditions)

    def to_table(
        self,
        join_method: MergeHow = "outer",
        mask_missing: bool = False,
        na_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Merge individual experimental conditions to one table.
        You might use this method to extract a Pandas data frame from
        the dataset and keep working using common procedures.

        Note that the entry identifier is in the index of the
        data frame.

        By default, values are returned as stored, so missing values may
        appear as ``0``. Generic tools (e.g., ``DataFrame.mean``,
        ``DataFrame.corr``, scikit-learn) treat such zeros as measured
        values. Set ``mask_missing=True`` to get missing values as ``NaN``
        instead, which pandas skips by default.

        Parameters
        ----------
        join_method : MergeHow, optional
            Method of joining records of each experimental condition in the output.
            Records absent from a condition are ``NaN`` in its columns.
        mask_missing : bool, optional
            If ``True``, values below or equal to ``na_threshold`` are
            replaced by ``NaN``. Default is ``False``.
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.
            Used only if ``mask_missing`` is ``True``.

        Returns
        -------
        pd.DataFrame
            A Pandas data frame containing all experimental conditions.

        See Also
        --------
        detection_matrix : Whether each record was detected in each experiment.

        Examples
        --------
        >>> table = dataset.to_table(mask_missing=True)
        >>> table.corr()  # zeros no longer count as measurements
        """
        tables = [
            c.to_table(mask_missing=mask_missing, na_threshold=na_threshold)
            for c in self._conditions
        ]
        return self._join_list_of_tables(tables, how=join_method)

    def detection_matrix(
        self, na_threshold: float = 0.0, join_method: MergeHow = "outer"
    ) -> pd.DataFrame:
        """
        Return whether each record was detected in each experiment.

        A value is detected if it is above ``na_threshold``. Missing
        values, stored as ``0`` or ``NaN``, are not detected. Neither are
        records absent from a condition after joining.

        Useful for presence/absence analyses, such as records detected in
        one condition but not another, or patterns of missing values.
        To count detections, use :meth:`frequency` instead.

        Parameters
        ----------
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.
        join_method : MergeHow, optional
            Method of joining records of each experimental condition in the
            output.

        Returns
        -------
        pd.DataFrame
            A boolean data frame with the same index and columns as
            :meth:`to_table` with the same ``join_method``.

        Examples
        --------
        >>> detected = dataset.detection_matrix()
        >>> only_in_treated = detected[
        ...     detected[dataset.experiment_names("treated")].any(axis=1)
        ...     & ~detected[dataset.experiment_names("control")].any(axis=1)
        ... ].index
        """
        return self._detection_mask(self.to_table(join_method=join_method), na_threshold)

    def correlation(
        self,
        method: CorrelationMethod = "spearman",
        pairs: ExperimentPairs | Sequence[tuple[str, str]] = "within_conditions",
        missing: MissingHandling = "pairwise",
        log_transform: bool | None = None,
        min_records: int = 10,
        na_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Correlation coefficient between pairs of experiments.

        Use it to check how reproducible the experiments are: replicates of
        the same condition should correlate more closely than experiments
        of different conditions.

        Missing values (values below or equal to ``na_threshold``, stored as
        ``0`` or ``NaN``) are never treated as measurements.

        The choice of coefficient matters:

        * ``"spearman"`` (default) and ``"kendall"`` are rank-based. They
          are robust to outliers, and a log transformation does not change
          them.
        * ``"pearson"`` depends on the scale of the values. Intensities
          span several orders of magnitude, so on raw values a few dozen of
          the most abundant records dominate the coefficient. For this
          reason, values are log2 transformed by default before computing
          it (see ``log_transform``).

        Parameters
        ----------
        method : {"spearman", "pearson", "kendall"}, optional
            Correlation coefficient. Default is ``"spearman"``.
        pairs : {"within_conditions", "across_conditions", "all"} or list of tuples, optional
            Which pairs of experiments to correlate.
            ``"within_conditions"`` (default): pairs of experiments of the
            same condition (e.g., biological replicates).
            ``"across_conditions"``: pairs of experiments of different
            conditions. ``"all"``: every pair of experiments.
            You can also pass a list of ``(experiment_a, experiment_b)``
            tuples.
        missing : {"pairwise", "listwise"}, optional
            Which records to use.
            ``"pairwise"`` (default): for each pair, the records with a value
            in both experiments. ``"listwise"``: only the records with a
            value in every experiment that is part of a selected pair, so
            all pairs are computed on the same records.
        log_transform : bool or None, optional
            Log2 transform the values before computing the coefficient.
            ``None`` (default) means ``True`` for ``"pearson"`` and ``False``
            for the rank-based methods, which a log transformation does not
            change. Set to ``False`` if the dataset is already
            log-transformed.
        min_records : int, optional
            Minimum number of records used for a pair. Pairs with fewer
            records get a ``NaN`` coefficient. Default is ``10``.
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.

        Returns
        -------
        pd.DataFrame
            One row per pair, with columns ``experiment_a``,
            ``experiment_b``, ``condition_a``, ``condition_b``,
            ``coefficient`` and ``n_records`` (the number of records used).
            For the named selections, each pair appears once, with
            ``experiment_a`` before ``experiment_b`` in the dataset's
            experiment order.

        Raises
        ------
        ValueError
            If an argument has an invalid value, or if ``log_transform`` is
            used and some values to transform are not positive.

        Warns
        -----
        UserWarning
            If the values are log-transformed by this method, but they seem
            to be log-transformed already.

        See Also
        --------
        bland_altman : Bias and limits of agreement between pairs of
            experiments.
        omicspylib.plots.plot_correlation_heatmap : Heatmap of all pairs.

        Examples
        --------
        >>> corr = dataset.correlation()  # doctest: +SKIP
        >>> corr.groupby("condition_a")["coefficient"].median()  # doctest: +SKIP

        Get a matrix of all pairs, e.g., to plot a heatmap:

        >>> corr = dataset.correlation(pairs="all")  # doctest: +SKIP
        >>> corr.pivot(
        ...     index="experiment_a", columns="experiment_b", values="coefficient"
        ... )  # doctest: +SKIP
        """
        if log_transform is None:
            log_transform = method == "pearson"

        table, exp_pairs = self._paired_values(pairs, missing, log_transform, na_threshold)
        result = calc_pairwise_correlation(
            table,
            pairs=[(exp_a, exp_b) for exp_a, _, exp_b, _ in exp_pairs],
            method=method,
            min_records=min_records,
        )
        return self._add_pair_conditions(result, exp_pairs)

    def bland_altman(
        self,
        pairs: ExperimentPairs | Sequence[tuple[str, str]] = "within_conditions",
        missing: MissingHandling = "pairwise",
        log_transform: bool = True,
        min_records: int = 10,
        na_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Bland-Altman agreement between pairs of experiments.

        For each pair, the difference between the two experiments is
        calculated for every record with a value in both. The mean of the
        differences (``bias``) shows a systematic offset between the
        experiments, e.g., a loading or normalization difference. The limits
        of agreement (``bias`` ± 1.96 standard deviations) contain about 95%
        of the differences, and show how far apart two measurements of the
        same record typically are.

        Unlike a correlation coefficient, these values are in the units of
        the data. On log2 values (default), the difference is a log2 fold
        change: a ``bias`` of ``1`` means that ``experiment_a`` is two
        times higher than ``experiment_b`` on average.

        Missing values (values below or equal to ``na_threshold``, stored as
        ``0`` or ``NaN``) are never treated as measurements.

        Parameters
        ----------
        pairs : {"within_conditions", "across_conditions", "all"} or list of tuples, optional
            Which pairs of experiments to compare. See :meth:`correlation`.
            You can also pass a list of ``(experiment_a, experiment_b)``
            tuples.
        missing : {"pairwise", "listwise"}, optional
            Which records to use. See :meth:`correlation`.
        log_transform : bool, optional
            Log2 transform the values first. Default is ``True``, because the
            differences between raw intensities grow with the intensity. Set
            to ``False`` if the dataset is already log-transformed.
        min_records : int, optional
            Minimum number of records used for a pair. Pairs with fewer
            records get ``NaN`` values. Default is ``10``.
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.

        Returns
        -------
        pd.DataFrame
            One row per pair, with columns ``experiment_a``,
            ``experiment_b``, ``condition_a``, ``condition_b``, ``bias``
            (mean of ``experiment_a - experiment_b``), ``sd`` (standard
            deviation of the differences), ``loa_lower``, ``loa_upper``
            (limits of agreement) and ``n_records``.

        See Also
        --------
        correlation : Correlation coefficient between pairs of experiments.
        omicspylib.plots.plot_experiment_pair : Bland-Altman or scatter plot
            of one pair.

        Examples
        --------
        >>> agreement = dataset.bland_altman()  # doctest: +SKIP
        >>> agreement[agreement["bias"].abs() > 0.5]  # doctest: +SKIP
        """
        table, exp_pairs = self._paired_values(pairs, missing, log_transform, na_threshold)
        result = calc_bland_altman(
            table,
            pairs=[(exp_a, exp_b) for exp_a, _, exp_b, _ in exp_pairs],
            min_records=min_records,
        )
        return self._add_pair_conditions(result, exp_pairs)

    def _paired_values(
        self,
        pairs: ExperimentPairs | Sequence[tuple[str, str]],
        missing: MissingHandling,
        log_transform: bool,
        na_threshold: float,
    ) -> tuple[pd.DataFrame, list[tuple[str, str, str, str]]]:
        """
        The values of the experiments in the selected pairs, ready for a
        pairwise comparison, and the pairs as ``(experiment_a, condition_a,
        experiment_b, condition_b)``.

        Missing values are ``NaN``. With ``missing="listwise"``, only the
        records with a value in every experiment of the pairs are kept.
        """
        if missing not in ("pairwise", "listwise"):
            raise ValueError(f"missing must be 'pairwise' or 'listwise', got {missing!r}.")

        exp_pairs = self._experiment_pairs(pairs)
        involved = list(
            dict.fromkeys(e for exp_a, _, exp_b, _ in exp_pairs for e in (exp_a, exp_b))
        )
        table = self.to_table(join_method="outer", mask_missing=True, na_threshold=na_threshold)
        table = table.loc[:, involved]
        if missing == "listwise":
            table = table.dropna()
        if log_transform:
            table = self._log2_masked(table)
        return table, exp_pairs

    @staticmethod
    def _add_pair_conditions(
        result: pd.DataFrame, exp_pairs: list[tuple[str, str, str, str]]
    ) -> pd.DataFrame:
        result.insert(2, "condition_a", [cond_a for _, cond_a, _, _ in exp_pairs])
        result.insert(3, "condition_b", [cond_b for _, _, _, cond_b in exp_pairs])
        return result

    def _experiment_pairs(
        self, pairs: ExperimentPairs | Sequence[tuple[str, str]]
    ) -> list[tuple[str, str, str, str]]:
        """
        Pairs of experiments as ``(experiment_a, condition_a, experiment_b,
        condition_b)``. Named selections follow the dataset's experiment order.
        """
        experiments = [
            (exp, cond) for cond in self.condition_names for exp in self.experiment_names(cond)
        ]
        if not isinstance(pairs, str):
            exp_to_cond = dict(experiments)
            explicit = []
            for pair in pairs:
                if len(pair) != 2:
                    raise ValueError(f"Each pair must have two experiment names, got {pair!r}.")
                exp_a, exp_b = pair
                unknown = [e for e in (exp_a, exp_b) if e not in exp_to_cond]
                if unknown:
                    raise ValueError(
                        f"Experiments not found: {unknown}. "
                        f"Available experiments: {list(exp_to_cond)}."
                    )
                if exp_a == exp_b:
                    raise ValueError(f"A pair needs two different experiments, got {pair!r}.")
                explicit.append((exp_a, exp_to_cond[exp_a], exp_b, exp_to_cond[exp_b]))
            return explicit

        if pairs not in ("within_conditions", "across_conditions", "all"):
            raise ValueError(
                "pairs must be 'within_conditions', 'across_conditions', 'all' "
                f"or a list of (experiment_a, experiment_b) tuples, got {pairs!r}."
            )
        selected = []
        for i, (exp_a, cond_a) in enumerate(experiments):
            for exp_b, cond_b in experiments[i + 1:]:
                same_condition = cond_a == cond_b
                if (
                    pairs == "all"
                    or (pairs == "within_conditions" and same_condition)
                    or (pairs == "across_conditions" and not same_condition)
                ):
                    selected.append((exp_a, cond_a, exp_b, cond_b))
        return selected

    @staticmethod
    def _log2_masked(table: pd.DataFrame) -> pd.DataFrame:
        """Log2 of the masked values, with checks for invalid or already logged values."""
        values = table.to_numpy(dtype=float)
        observed = values[~np.isnan(values)]
        if observed.size == 0:
            return table
        if (observed <= 0).any():
            raise ValueError(
                "Cannot log-transform values that are not positive. Use a "
                "non-negative na_threshold, or set log_transform=False."
            )
        if observed.max() < 100:
            warnings.warn(
                "The values seem to be log-transformed already (all below 100), "
                "but they are log-transformed again. If the dataset is already "
                "log-transformed, set log_transform=False.",
                category=UserWarning,
                stacklevel=4,
            )
        return cast(pd.DataFrame, np.log2(table))

    def missing_values(
        self, na_threshold: float = 0.0
    ) -> tuple[pd.DataFrame, int, int]:
        """
        .. deprecated:: 0.2.1
            `missing_values` is deprecated and will be removed in a future release.
            Use `record_counts(value_type="missing")` instead. NOTE that the `n_missing`
            column is renamed to `n_counts`.

        Returns the number of missing values per experiment and condition.
        Missing values are considered the cases that are either missing
        or are below the specified threshold.

        Parameters
        ----------
        na_threshold : float
            Values below or equal to this threshold are considered missing.

        Returns
        -------
        pd.DataFrame
            A Pandas data frame with the number of missing cases per
            experiment and condition.
        Int
            Number of missing values.
        Int
            Number of values in total
        """
        warnings.warn(
            "missing_values is deprecated and will be removed in a future release."
            " Use record_counts(value_type='missing') instead. NOTE that the `n_missing` column"
            "is renamed to `n_counts`.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        data = self.to_table()
        n_missing_per_exp = data.shape[0] - np.sum(self._detection_mask(data, na_threshold), axis=0)

        n_missing = n_missing_per_exp.reset_index()
        n_missing.columns = ["experiment", "n_missing"]

        # total number of missing values across all datasets.
        n_missing_total = np.sum(n_missing_per_exp)
        n_total = data.shape[0] * len(data.columns)
        names_lookup = [
            {"experiment": exp, "condition": cond.name}
            for cond in self._conditions
            for exp in cond.experiment_names
        ]
        names_lookup_df = pd.DataFrame(names_lookup)
        out_df = n_missing.merge(names_lookup_df, on="experiment", how="left")

        return out_df, n_missing_total, n_total

    def impute(
        self,
        method: ImputeMethod,
        na_threshold: float = 0.0,
        value: float | None = None,
        shift: float = 0.0,
        random_noise: bool = False,
    ) -> Self:
        """
        Impute missing values with any of the specified methods.
        Note that missing value imputation my introduce artifacts in
        the analysis. Consider the level of missing value imputation
        before interpreting your results.

        Parameters
        ----------
        method: str
            Imputation method. Can be one of:
                - ``fixed``: A fixed value. All values below the given threshold
                  will be set to that value. To use this method, you also need
                  to specify the ``value`` parameter.
                - ``global min|mean|median``: First the ``min|mean|median`` value of
                  the dataset is calculated, and then missing values are set
                  to that fixed value. You can also specify the `shift`
                  parameter to shift the calculated min by a fixed step.
                - ``global row min|mean|median``: Similar to ``global min`` but the
                  min|median|mean value refers to the row entry value instead of
                  the value across all entries of that table.
                - ``group row min|mean|median``. Similar to the previous but
                  now the min|mean|median is based on the values of the group.
        na_threshold: float
            Values below or equal to this threshold are considered missing.
        value: float, optional
            If ``fixed`` method is specified, you also need to set that value here.
        shift: float
            If ``global|group-min`` method is specified, you can also decrease
            that value by a fixed step.
        random_noise: bool, optional
            If specified random noise based on the global or within group variability,
            will be added. Imputed values will be selected from a normal distribution
            with mean the selected value (depending on the method) and std the within
            group or global standard deviation (depending on the method). Because you
            draw random values from a normal distribution, consider transforming your
            data if needed, to approximate it (e.g., apply log2 transformation, if needed).
            After imputation, you can back_transform to the original scale.
        """
        imputed_conditions = copy.deepcopy(self._conditions)
        if method == "fixed":
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions,
                "fixed",
                na_threshold,
                value,
                random_noise=random_noise,
            )
        elif method == "global min":
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions,
                "min",
                na_threshold,
                shift,
                random_noise=random_noise,
            )
        elif method == "global mean":
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions,
                "mean",
                na_threshold,
                shift,
                random_noise=random_noise,
            )
        elif method == "global median":
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions,
                "median",
                na_threshold,
                shift,
                random_noise=random_noise,
            )
        elif method == "global row min":
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions,
                "min",
                na_threshold,
                shift,
                random_noise=random_noise,
            )
        elif method == "global row mean":
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions,
                "mean",
                na_threshold,
                shift,
                random_noise=random_noise,
            )
        elif method == "global row median":
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions,
                "median",
                na_threshold,
                shift,
                random_noise=random_noise,
            )
        elif method == "group row min":
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions,
                "row min",
                na_threshold,
                value,
                shift,
                random_noise=random_noise,
            )
        elif method == "group row mean":
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions,
                "row mean",
                na_threshold,
                value,
                shift,
                random_noise=random_noise,
            )
        elif method == "group row median":
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions,
                "row median",
                na_threshold,
                value,
                shift,
                random_noise=random_noise,
            )

        return self.__class__(conditions=imputed_conditions)

    def _impute_by_global_row_value(
        self, imputed_conditions, value_type, na_threshold, shift, random_noise
    ) -> list[TabularExperimentalConditionDataset]:
        df = self.to_table()
        df = self._mask_missing(df, na_threshold)
        if value_type == "min":
            values_series = df.min(axis=1, skipna=True)
        elif value_type == "mean":
            values_series = df.mean(axis=1, skipna=True)
        elif value_type == "median":
            values_series = df.median(axis=1, skipna=True)
        else:
            raise ValueError(f"Value type {value_type} not supported.")

        return [
            c.impute(
                method="fixed row",
                na_threshold=na_threshold,
                value=values_series - shift,
                random_noise=random_noise,
            )
            for c in imputed_conditions
        ]

    def _impute_by_global_value(
        self,
        imputed_conditions,
        value_type,
        na_threshold,
        shift,
        random_noise: bool = False,
    ) -> list[TabularExperimentalConditionDataset]:
        df = self.to_table()
        df = self._mask_missing(df, na_threshold)
        all_values = df.to_numpy().flatten()
        if value_type == "min":
            targ_value = np.nanmin(all_values)
        elif value_type == "mean":
            targ_value = np.nanmean(all_values)
        elif value_type == "median":
            targ_value = np.nanmedian(all_values)
        else:
            raise ValueError(f"Value type {value_type} not supported.")

        return self._intra_group_imputation(
            conditions=imputed_conditions,
            method="fixed",
            na_threshold=na_threshold,
            value=targ_value - shift,
            random_noise=random_noise,
        )

    @staticmethod
    def _intra_group_imputation(
        conditions,
        method,
        na_threshold,
        value,
        shift: float = 0.0,
        random_noise: bool = False,
    ) -> list[TabularExperimentalConditionDataset]:
        return [
            c.impute(
                method=method,
                na_threshold=na_threshold,
                value=value,
                shift=shift,
                random_noise=random_noise,
            )
            for c in conditions
        ]

    def normalize(
        self,
        method: NormMethod,
        ref_experiment: str | Sequence[str] | None = None,
        ref_condition: str | None = None,
        use_common_records: bool = False,
        na_threshold: float = 0.0,
        trim_fraction: float = 0.0,
        within_conditions: bool = False,
        *,
        ref_exp: str | None = None,
    ) -> Self:
        """
        Normalize the dataset.
        Any required transformations must be done before calling the
        ``normalize`` method. For clarity, this is not handled internally.

        For example, you might use ``log2_transform`` method first then
        ``normalize`` and finally ``log2_backtransform`` the normalized
        values to return to the same units.

        Each experiment is shifted so that its central value matches the
        central value of a reference experiment. The central value is the
        ``mean`` (optionally trimmed, see ``trim_fraction``) or the
        ``median`` of the experiment's intensities, depending on ``method``.
        The median is less sensitive to outliers and skewed distributions.

        Steps, for either method:

        - Without common records (default):
            1. Pick the reference experiment: ``ref_experiment`` if given, else the
               experiment with the most records (within ``ref_condition``
               if given, otherwise across the whole dataset).
            2. Calculate the mean or median intensity of each experiment,
               using all of its non-missing values, and the difference from
               the reference.
            3. Shift each experiment's intensity by that difference.
        - With common records (``use_common_records=True``):
            1. Pick the reference experiment as above.
            2. For each other experiment, in a pairwise comparison with the
               reference:

               i. Keep only the records present in both experiments.
               ii. Calculate the mean or median of both on those records,
                   and the difference from the reference.
               iii. Shift all intensities of that experiment by that
                    difference.

        With ``within_conditions=True`` each condition is normalized on its
        own, and the differences between conditions are kept. Within a
        condition, the replicates are shifted to:

        - the central value of the condition's reference experiment, if one
          is given in ``ref_experiment``. With ``use_common_records=True``
          each replicate is compared pairwise with the reference, as above.
        - otherwise, the mean of the replicates' central values, so the
          overall level of the condition does not change. With
          ``use_common_records=True`` the central values are calculated on
          the records detected in all replicates of the condition.

        Use it only when the differences in overall level between conditions
        are expected to be biological rather than technical, e.g. a pulldown
        against an IgG control, or fractions with different total amounts.
        Otherwise, technical offsets between conditions (e.g. from separate
        sample preparation batches) are kept and show up as global up- or
        down-regulation.

        Examples
        --------
        >>> norm = dataset.log2_transform() \\
        ...     .normalize(method="median") \\
        ...     .log2_backtransform()  # doctest: +SKIP

        Normalize the replicates of each condition to their mean, using
        ``c1_rep1`` as the reference of condition ``c1``:

        >>> norm = dataset.log2_transform() \\
        ...     .normalize(method="median", within_conditions=True,
        ...                ref_experiment=["c1_rep1"]) \\
        ...     .log2_backtransform()  # doctest: +SKIP

        Parameters
        ----------
        method:
            Normalization method: ``"mean"`` or ``"median"``.
        ref_experiment: str or list of str, optional
            If specified, this experiment will be considered the reference.
            If this is set, the ``ref_condition`` field is ignored.
            With ``within_conditions=True`` it may be a list, with at most
            one experiment per condition. Each one is the reference of its
            own condition; conditions without one are normalized to the
            mean of their replicates.
        ref_condition: str, optional
            If specified, the experiment of that condition with the most records
            will be considered the reference.
            Note that ``ref_experiment`` should not be set.
            Not allowed with ``within_conditions=True``.
        use_common_records: bool
            If set to ``True``, common records, in a pairwise comparison with the
            reference, will be considered for normalization.
        na_threshold: float
            Values below or equal to this threshold are considered missing.
        trim_fraction: float
            Use a trimmed mean instead of a plain mean, discarding a fraction
            of the most extreme values from the top and bottom ends. Applies
            both when normalizing to all records and to shared records.
            Each experiment is trimmed independently. For example, if
            ``trim_fraction`` is set to ``0.1``, a total of 10% of the values
            will be discarded (5% from each edge). Must be in ``[0, 1)``.
            Only used with ``method="mean"``; it has no effect on the
            median, which symmetric trimming does not change.
        within_conditions: bool
            If set to ``True``, normalize the replicates of each condition
            among themselves only, keeping the differences between
            conditions. See above.

        ref_exp: str, optional
            .. deprecated:: 0.3.0
                Use ``ref_experiment`` instead.

        Returns
        -------
        A new instance of the same object with normalized values.
        """
        ref_experiment = resolve_renamed_arg(
            ref_experiment, ref_exp, "ref_experiment", "ref_exp"
        )
        if not 0.0 <= trim_fraction < 1.0:
            raise ValueError(
                f"trim_fraction must be in the range [0, 1), got {trim_fraction}."
            )

        if method not in ("mean", "median"):
            raise NotImplementedError(
                f"Normalization method {method} is not implemented."
            )

        if within_conditions:
            if ref_condition is not None:
                raise ValueError(
                    "ref_condition cannot be used with within_conditions=True. "
                    "Use ref_experiment to set a reference per condition."
                )
            refs = self._select_within_condition_refs(ref_experiment)
            exp_conditions = copy.deepcopy(self._conditions)
            for condition in exp_conditions:
                self._normalize_condition(
                    condition,
                    ref_experiment=refs.get(condition.name),
                    na_threshold=na_threshold,
                    method=method,
                    trim_fraction=trim_fraction,
                    use_common_records=use_common_records,
                )
            return self.__class__(conditions=exp_conditions)

        if ref_experiment is not None and not isinstance(ref_experiment, str):
            raise TypeError(
                "A list of reference experiments is only allowed with "
                "within_conditions=True. Pass a single experiment name."
            )

        ref_experiment = self._select_norm_ref(na_threshold, ref_condition, ref_experiment)

        # step 2 - calculate difference
        if use_common_records:
            exp_conditions = self._norm_with_shared_records(
                na_threshold, ref_experiment, method=method, trim_fraction=trim_fraction
            )
        else:
            exp_conditions = self._base_normalization(
                na_threshold, ref_experiment, method=method, trim_fraction=trim_fraction
            )

        return self.__class__(conditions=exp_conditions)

    def _norm_with_shared_records(
        self,
        na_threshold: float,
        ref_experiment: str,
        method: NormMethod = "mean",
        trim_fraction: float = 0.0,
    ) -> Sequence[TabularExperimentalConditionDataset]:
        exp_conditions_data = copy.deepcopy(self._conditions)

        ref_c = next(
            (c for c in exp_conditions_data if ref_experiment in c.experiment_names), None
        )
        if ref_c:
            # The reference experiment is never shifted, so its values
            # can be read once, up front.
            ref_case = ref_c.filter(experiments=ref_experiment).to_table()
            for exp_c in exp_conditions_data:
                for targ_exp in exp_c.experiment_names:
                    if targ_exp == ref_experiment:
                        continue
                    targ_case = exp_c.filter(experiments=targ_exp).to_table()
                    df = ref_case.merge(
                        targ_case, left_index=True, right_index=True, how="inner"
                    )
                    df = self._mask_missing(df, na_threshold)
                    df = df.dropna()

                    center_ref = _central_value(df[ref_experiment], method, trim_fraction)
                    center_targ = _central_value(df[targ_exp], method, trim_fraction)
                    norm_diff = center_targ - center_ref
                    exp_c.shift(targ_exp, norm_diff, na_threshold=na_threshold)

        return exp_conditions_data

    def _base_normalization(
        self,
        na_threshold: float,
        ref_experiment: str,
        method: NormMethod = "mean",
        trim_fraction: float = 0.0,
    ) -> Sequence[TabularExperimentalConditionDataset]:
        centers: dict[str, float] = {}
        for c in self._conditions:
            for exp in c.experiment_names:
                values = c.filter(experiments=exp).to_table()[exp]
                centers[exp] = _central_value(
                    values[self._detection_mask(values, na_threshold)], method, trim_fraction
                )
        ref_center = centers[ref_experiment]
        exp_conditions_data = copy.deepcopy(self._conditions)
        for condition in exp_conditions_data:
            for exp in condition.experiment_names:
                shift_value = centers[exp] - ref_center
                if shift_value != 0:
                    condition.shift(exp, value=shift_value, na_threshold=na_threshold)
        return exp_conditions_data

    def _select_norm_ref(self, na_threshold, ref_condition, ref_experiment) -> str:
        """Select a reference experiment for the normalization step."""
        if ref_experiment is not None:
            available_exps = self.experiment_names()
            if ref_experiment not in available_exps:
                raise ValueError(
                    f"Reference experiment {ref_experiment!r} not found. "
                    f"Available experiments: {available_exps}."
                )
        elif ref_condition is not None:
            if ref_condition not in self.condition_names:
                raise ValueError(
                    f"Reference condition {ref_condition!r} not found. "
                    f"Available conditions: {self.condition_names}."
                )
            n_entries_per_exp = self.frequency(
                na_threshold=na_threshold, axis=0, conditions=[ref_condition]
            ).transpose()
            ref_experiment = n_entries_per_exp["frequency"].idxmax()
        else:
            n_entries_per_exp = self.frequency(
                na_threshold=na_threshold, axis=0
            ).transpose()
            ref_experiment = n_entries_per_exp["frequency"].idxmax()

        return str(ref_experiment)

    def _select_within_condition_refs(
        self, ref_experiment: str | Sequence[str] | None
    ) -> dict[str, str]:
        """Map each condition to its reference experiment, if one is given."""
        if ref_experiment is None:
            return {}
        names = [ref_experiment] if isinstance(ref_experiment, str) else list(ref_experiment)
        exp_to_condition = {
            exp: c.name for c in self._conditions for exp in c.experiment_names
        }
        refs: dict[str, str] = {}
        for exp in names:
            if exp not in exp_to_condition:
                raise ValueError(
                    f"Reference experiment {exp!r} not found. "
                    f"Available experiments: {list(exp_to_condition)}."
                )
            condition = exp_to_condition[exp]
            if condition in refs:
                raise ValueError(
                    f"Condition {condition!r} has more than one reference "
                    f"experiment: {refs[condition]!r} and {exp!r}."
                )
            refs[condition] = exp
        return refs

    def _normalize_condition(
        self,
        condition: TabularExperimentalConditionDataset,
        ref_experiment: str | None,
        na_threshold: float,
        method: NormMethod,
        trim_fraction: float,
        use_common_records: bool,
    ) -> None:
        """
        Normalize the replicates of one condition among themselves, in place.

        Replicates are shifted to the central value of ``ref_experiment``,
        or to the mean of the replicates' central values if it is ``None``.
        """
        table = condition.to_table(mask_missing=True, na_threshold=na_threshold)

        def center(values: pd.Series) -> float:
            return _central_value(values, method, trim_fraction)

        shifts: dict[str, float] = {}
        if ref_experiment is not None and use_common_records:
            for exp in table.columns:
                if exp == ref_experiment:
                    continue
                pair = table[[ref_experiment, exp]].dropna()
                shifts[exp] = center(pair[exp]) - center(pair[ref_experiment])
        else:
            if use_common_records:
                table = table.dropna()
            centers = {exp: center(table[exp]) for exp in table.columns}
            target = (
                centers[ref_experiment]
                if ref_experiment is not None
                else float(np.mean(list(centers.values())))
            )
            shifts = {exp: value - target for exp, value in centers.items()}

        for exp, value in shifts.items():
            if np.isnan(value):
                raise ValueError(
                    f"Cannot normalize experiment {exp!r} of condition "
                    f"{condition.name!r}: no records to calculate its central "
                    "value from."
                )
            if value != 0:
                condition.shift(exp, value=value, na_threshold=na_threshold)
