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

AxisName = Literal["rows", "columns"]
NormMethod = Literal["mean"]  # future implementation for quantile, median etc
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


class TabularExperimentalConditionDataset(abc.ABC):
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
        return self._data.index.values.tolist()

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
            "n_records_per_experiment": np.sum(self._data.values > na_threshold, axis=0).tolist(),
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
        df = self._data.copy()
        df[df <= na_threshold] = np.nan
        if axis is None:
            min_value = np.nanmin(df.values.flatten())
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
            self._data > na_threshold, axis=0
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
        mask = self._data > na_threshold
        data = self._data.copy()
        data[~mask] = np.nan
        mean = data.sum(axis=axis) / mask.sum(axis=axis)
        if axis == 1:  # row mean
            return pd.DataFrame({f"mean_{self.name}": mean})
        else:  # column mean
            return pd.DataFrame({"mean": mean})

    @abc.abstractmethod
    def filter(
        self,
        exp: str | list | None = None,
        min_frequency: int | None = None,
        na_threshold: float = 0.0,
        ids: list | None = None,
    ) -> Self:
        raise NotImplementedError

    def _apply_filter(self, exp, min_frequency, na_threshold, ids=None) -> pd.DataFrame:
        data = self._data.copy()
        if min_frequency is not None:
            valid_rows = (data > na_threshold).sum(axis=1) >= min_frequency
            data = data.loc[valid_rows, :].copy()
        if isinstance(exp, str):
            exp = [exp]
        if exp is not None:
            # You might filter by providing names across experimental
            # conditions. So when you work on each condition separately,
            # some names are no longer valid.
            local_exp = [ex for ex in exp if ex in data.columns]
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
        f = np.sum(self._data > na_threshold, axis=axis)
        if axis == 1:
            return pd.DataFrame({f"frequency_{self.name}": f})
        else:
            return pd.DataFrame({"frequency": f})

    @abc.abstractmethod
    def drop(
        self,
        exp: str | list | None = None,
        ids: list | None = None,
        omit_missing_cols: bool = True,
        **kwargs,
    ) -> Self:
        raise NotImplementedError

    def _apply_drop(
        self,
        exp: str | list | None = None,
        ids: list | None = None,
        omit_missing_cols: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        data = self._data.copy()
        if isinstance(exp, str):
            exp = [exp]

        if omit_missing_cols and exp is not None:
            # allow the user to pass column names that don't exist or
            # are already excluded from previous steps.
            exp = [e for e in exp if e in data.columns]

        if exp is not None:
            data = data.drop(exp, axis=1)

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
        self._data[self._data <= na_threshold] = np.nan

        if random_noise:
            rand_noise_std = self._calc_mean_std()
        else:
            rand_noise_std = None

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
                        row.iloc[j] = np.random.normal(val.iloc[val_idx], std_value)
        # case where you fill with the same value across rows
        else:
            if std_value is None:
                row = row.fillna(val)
            else:
                for j, v in enumerate(row):
                    if np.isnan(v):
                        row.iloc[j] = np.random.normal(val, std_value)

        return row

    def to_table(self) -> pd.DataFrame:
        """
        Return the values of this condition as a pandas data frame.

        Returns
        -------
        pd.DataFrame
            A copy of the table, with record ids as the index and one column
            per experiment. Modifying it does not change this object.
        """
        return self._data.copy()

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
        mask = self._data[exp] > na_threshold
        self._data.loc[mask, exp] -= value


class TabularDataset(abc.ABC):
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
        assert axis in [0, 1]

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
            n_counts_per_exp = data.shape[0] - np.sum(data > na_threshold, axis=0)
        elif value_type == "present":
            n_counts_per_exp = np.sum(data > na_threshold, axis=0)
        else:
            raise ValueError("type must be either 'missing' or 'present'")  # pyright: ignore

        n_counts_df = n_counts_per_exp.reset_index()
        n_counts_df.columns = ["experiment", "n_counts"]

        # total number of missing values across all datasets.
        n_counts_total = np.sum(n_counts_per_exp)
        n_total = data.shape[0] * len(data.columns)
        names_lookup = []
        for cond in self._conditions:
            experiments = cond.experiment_names
            for exp in experiments:
                names_lookup.append({"experiment": exp, "condition": cond.name})
        names_lookup_df = pd.DataFrame(names_lookup)
        out_df = n_counts_df.merge(names_lookup_df, on="experiment", how="left")

        return out_df, n_counts_total, n_total

    def drop(
        self,
        exp: str | list | None = None,
        cond: str | list | None = None,
        ids: list | None = None,
    ) -> Self:
        """
        Drop specified experiment(s) and or condition(s).

        Parameters
        ----------
        exp: str, list, optional
            Experiment name(s) to be dropped.
        cond: str, list, optional
            Experimental condition(s) to be dropped.
        ids: list, optional
            If specified, a list of records ids to drop from the dataset.

        Returns
        -------
        Self
            An object of the same instance type without
            the specified experiment(s) and/or condition(s).
        """
        filt_conditions = copy.deepcopy(self._conditions)
        if isinstance(exp, str):
            exp = [exp]
        if isinstance(cond, str):
            cond = [cond]
        if cond is not None:
            filt_conditions = [c for c in filt_conditions if c.name not in cond]

        if exp is not None:
            filt_conditions = [c.drop(exp) for c in filt_conditions]

        if ids is not None:
            filt_conditions = [c.drop(ids=ids) for c in filt_conditions]

        return self.__class__(conditions=filt_conditions)

    def filter(
        self,
        exp: str | list | None = None,
        cond: list | None = None,
        min_frequency: int | None = None,
        na_threshold: float = 0.0,
        ids: list | None = None,
    ) -> Self:
        """
        Filter the dataset based on a given set of properties.

        Parameters
        ----------
        exp: list, str, optional
            List of experiment name or a single experiment to keep.
            Leave empty to keep all experiments.
        cond: list, optional
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

        Returns
        -------
        ProteinsDataset
            A new instance of the dataset object, filtered based on the
            user's input.
        """
        exp_conditions = list(self._conditions).copy()
        if isinstance(exp, str):
            exp = [exp]
        if isinstance(cond, str):
            cond = [cond]

        if cond is not None:
            exp_conditions = [c for c in exp_conditions if c.name in cond]

        filt_min_f = min_frequency is not None
        filt_na_th = na_threshold is not None
        filt_ids = ids is not None
        if filt_min_f or filt_na_th or filt_ids:
            exp_conditions = [
                c.filter(
                    exp=exp,
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
            lambda left, right: pd.merge(
                left, right, left_index=True, right_index=True, how=how
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

    def to_table(self, join_method: MergeHow = "outer") -> pd.DataFrame:
        """
        Merge individual experimental conditions to one table.
        You might use this method to extract a Pandas data frame from
        the dataset and keep working using common procedures.

        Note that the entry identifier is in the index of the
        data frame.

        Parameters
        ----------
        join_method: MergeHow, optional
            Method of joining records of each experimental condition in the output.

        Returns
        -------
        pd.DataFrame
            A Pandas data frame containing all experimental conditions.
        """
        tables = [c.to_table() for c in self._conditions]
        return self._join_list_of_tables(tables, how=join_method)

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
        n_missing_per_exp = data.shape[0] - np.sum(data > na_threshold, axis=0)

        n_missing = n_missing_per_exp.reset_index()
        n_missing.columns = ["experiment", "n_missing"]

        # total number of missing values across all datasets.
        n_missing_total = np.sum(n_missing_per_exp)
        n_total = data.shape[0] * len(data.columns)
        names_lookup = []
        for cond in self._conditions:
            experiments = cond.experiment_names
            for exp in experiments:
                names_lookup.append({"experiment": exp, "condition": cond.name})
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
        df[df <= na_threshold] = np.nan
        if value_type == "min":
            values_series = df.min(axis=1, skipna=True)
        elif value_type == "mean":
            values_series = df.mean(axis=1, skipna=True)
        elif value_type == "median":
            values_series = df.median(axis=1, skipna=True)
        else:
            raise ValueError(f"Value type {value_type} not supported.")

        imputed_conditions = [
            c.impute(
                method="fixed row",
                na_threshold=na_threshold,
                value=values_series - shift,
                random_noise=random_noise,
            )
            for c in imputed_conditions
        ]
        return imputed_conditions

    def _impute_by_global_value(
        self,
        imputed_conditions,
        value_type,
        na_threshold,
        shift,
        random_noise: bool = False,
    ) -> list[TabularExperimentalConditionDataset]:
        df = self.to_table()
        df[df <= na_threshold] = np.nan
        all_values = df.values.flatten()
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
        imputed_conditions = [
            c.impute(
                method=method,
                na_threshold=na_threshold,
                value=value,
                shift=shift,
                random_noise=random_noise,
            )
            for c in conditions
        ]
        return imputed_conditions

    def normalize(
        self,
        method: NormMethod,
        ref_exp: str | None = None,
        ref_condition: str | None = None,
        use_common_records: bool = False,
        na_threshold: float = 0.0,
        trim_fraction: float = 0.0,
    ) -> Self:
        """
        Normalize the dataset.
        Any required transformations must be done before calling the
        ``normalize`` method. For clarity, this is not handled internally.

        For example, you might use ``log2_transform`` method first then
        ``normalize`` and finally ``log2_backtransform`` the normalized
        values to return to the same units.

        Normalization methods:

        - Mean without a use of common records without a ``ref_exp``.:
            1. Find the experiment with the most records and consider reference.
            2. Calculate mean experiment intensity and the difference from reference.
            3. Shift each experiment's intensity by the difference with reference.
        - mean without a use of common records with ref exp.:
            - Like above, but the reference experiment is defined by the user.
        - Mean with common records without a ref exp.:
            1. Find experiment with the most records and consider reference.
            2. Perform a pairwise comparison of each experiment with the reference where:
               i.
               Filter on common records.
               Ii.
               Calculate difference from reference.
               Iii.
               Shift all intensities of that experiment based on that difference.
        - mean with common records with a ref exp.:
            - Similar with previous (pairwise comparison and select common records),
              but reference is defined by the user.
        - mean with common records with a ref condition:
            - Similar with the previous, reference is selected automatically,
              from the condition specified.


        Parameters
        ----------
        method:
            Normalization method.
            For the moment, only normalization to the ``mean`` is supported.
        ref_exp: str, optional
            If specified, this experiment will be considered the reference.
            If this is set, the ``ref_condition`` field is ignored.
        ref_condition: str, optional
            If specified, the experiment of that condition with the most records
            will be considered the reference.
            Note that ``ref_exp`` should not be set.
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

        Returns
        -------
        A new instance of the same object with normalized values.
        """
        if not 0.0 <= trim_fraction < 1.0:
            raise ValueError(
                f"trim_fraction must be in the range [0, 1), got {trim_fraction}."
            )

        ref_exp = self._select_norm_ref(na_threshold, ref_condition, ref_exp)

        # step 2 - calculate difference
        if method == "mean" and use_common_records:
            exp_conditions = self._mean_norm_with_shared_records(
                na_threshold, ref_exp, trim_fraction=trim_fraction
            )
        elif method == "mean" and not use_common_records:
            exp_conditions = self._base_mean_normalization(
                na_threshold, ref_exp, trim_fraction=trim_fraction
            )
        else:
            raise NotImplementedError(
                f"Normalization method {method} with the specified "
                f"arguments is not implemented."
            )

        return self.__class__(conditions=exp_conditions)

    def _mean_norm_with_shared_records(
        self, na_threshold: float, ref_exp: str, trim_fraction: float = 0.0
    ) -> Sequence[TabularExperimentalConditionDataset]:
        exp_conditions_data = copy.deepcopy(self._conditions)

        ref_c = next(
            (c for c in exp_conditions_data if ref_exp in c.experiment_names), None
        )
        if ref_c:
            # The reference experiment is never shifted, so its values
            # can be read once, up front.
            ref_case = ref_c.filter(exp=ref_exp).to_table()
            for exp_c in exp_conditions_data:
                for targ_exp in exp_c.experiment_names:
                    if targ_exp == ref_exp:
                        continue
                    targ_case = exp_c.filter(exp=targ_exp).to_table()
                    df = ref_case.merge(
                        targ_case, left_index=True, right_index=True, how="inner"
                    )
                    df[df <= na_threshold] = np.nan
                    df = df.dropna()

                    mean_ref = _trimmed_mean(df[ref_exp], trim_fraction)
                    mean_targ = _trimmed_mean(df[targ_exp], trim_fraction)
                    norm_diff = mean_targ - mean_ref
                    exp_c.shift(targ_exp, norm_diff, na_threshold=na_threshold)

        return exp_conditions_data

    def _base_mean_normalization(
        self, na_threshold: float, ref_exp: str, trim_fraction: float = 0.0
    ) -> Sequence[TabularExperimentalConditionDataset]:
        if trim_fraction > 0.0:
            exp_means = {}
            for c in self._conditions:
                for exp in c.experiment_names:
                    values = c.filter(exp=exp).to_table()[exp]
                    exp_means[exp] = _trimmed_mean(
                        values[values > na_threshold], trim_fraction
                    )
            mean_before = pd.DataFrame([exp_means])
        else:
            mean_before = self.mean(na_threshold=na_threshold, axis=0)
        ref_mean = mean_before[ref_exp].values[0]
        mean_diff = mean_before - ref_mean
        exp_names = mean_diff.columns
        shift_values = mean_diff.values.reshape(-1)
        exp_conditions_data = copy.deepcopy(self._conditions)
        for condition in exp_conditions_data:
            for exp, shift_value in zip(exp_names, shift_values, strict=True):
                if exp in condition.experiment_names and shift_value != 0:
                    condition.shift(exp, value=shift_value, na_threshold=na_threshold)
        return exp_conditions_data

    def _select_norm_ref(self, na_threshold, ref_condition, ref_exp) -> str:
        """Select a reference experiment for the normalization step."""
        if ref_exp is not None:
            assert ref_exp in self.experiment_names(), (
                f"Reference experiment {ref_exp} not found."
            )
        elif ref_condition is not None:
            assert ref_condition in self.condition_names, (
                f"Reference condition {ref_condition} not found."
            )
            n_entries_per_exp = self.frequency(
                na_threshold=na_threshold, axis=0, conditions=[ref_condition]
            ).transpose()
            ref_exp = n_entries_per_exp["frequency"].idxmax()
        else:
            n_entries_per_exp = self.frequency(
                na_threshold=na_threshold, axis=0
            ).transpose()
            ref_exp = n_entries_per_exp["frequency"].idxmax()

        return str(ref_exp)
