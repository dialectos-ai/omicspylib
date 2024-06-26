from __future__ import annotations
import abc
from typing import List, Optional, Union, Literal, Type, TypeVar, Tuple
from functools import reduce

import numpy as np
import pandas as pd


AxisName = Literal['rows', 'columns']
MergeHow = Literal['left', 'right', 'inner', 'outer', 'cross']
ConditionImputeMethod = Literal[
    'fixed',
    'fixed row',
    'row min',
    'row mean',
    'row median'
]

T = TypeVar('T', bound='TabularDataset')


class TabularDatasetExpCondition(abc.ABC):
    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 id_col: str,
                 experiment_cols: list,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        name : str
            The name of the instance.
        data : pandas.DataFrame
            The input data for the instance.
        id_col : str
            The name of the column containing unique identifiers in the data.
        experiment_cols : list
            The list of column names representing different experiments in the data.
        """
        self._name = name
        self._data = data[[id_col]+experiment_cols].copy().set_index(id_col)
        self._id_col = id_col

    @property
    def _experiments(self) -> list:
        return self._data.columns.tolist()

    @property
    def n_experiments(self) -> int:
        """
        Returns the number of experiments.

        Returns
        -------
            int: The number of experiments.
        """
        return len(self._data.columns)

    @property
    def experiments(self) -> List[str]:
        """
        Get the list of experiment names.

        Returns
        -------
        list
            A list of experiment names of that condition.
        """
        return self._experiments

    @property
    def record_ids(self) -> List[str]:
        """
        Returns a list of unique protein ids as they are provided by the user.
        """
        return self._data.index.values.tolist()

    @property
    def name(self) -> str:
        """
        Get experimental condition name (e.g. treated, untreated etc.).
        """
        return self._name

    def describe(self) -> dict:
        """
        Returns basic information about the dataset.
        """
        return {
            'name': self._name,
            'n_experiments': self.n_experiments,
            'n_records': len(self.record_ids),
            'experiment_names': self._data.columns.tolist(),
            'n_records_per_experiment': np.sum(self._data.values > 0, axis=0).tolist()
        }

    def min(self,
            na_threshold: float = 0.0,
            axis: Optional[AxisName] = None) -> Union[float, pd.Series]:
        """
        Calculate minimum value of that condition.
        By default, calculates min value from all experiments,
        """
        df = self._data.copy()
        df[df <= na_threshold] = np.nan
        if axis is None:
            min_value = np.nanmin(df.values.flatten())
        elif axis == 'rows':
            min_value = df.min(axis=0)
        else: # axis = columns
            min_value = df.min(axis=1)

        return min_value

    def missing_values(self, na_threshold: float = 0.0) -> Tuple[pd.DataFrame, int, int]:
        """
        Calculate number of missing values per experiment.

        Parameters
        ----------
        na_threshold : float, optional
            Values equal or below this threshold will be considered missing.

        Returns
        -------
        pd.DataFrame
            A pandas data frame with the number of missing values per experiment.
        int
            Number of missing values in total.
        int
            Number of total values of that condition.
        """
        n_missing_per_exp = self._data.shape[0] - np.sum(self._data > na_threshold, axis=0)
        n_missing_total = np.sum(n_missing_per_exp)
        total_values = self._data.shape[0] * self._data.shape[1]
        df = pd.DataFrame({
            'experiment': self._data.columns,
            'n_missing': n_missing_per_exp.tolist(),
            'condition': self._name
        })
        return df, int(n_missing_total), int(total_values)

    def log2_transform(self: Type[T]) -> T:
        self._data = np.log2(self._data + 1)  # type: ignore
        return self

    def log2_backtransform(self: Type[T]) -> T:
        self._data = 2 ** self._data - 1
        return self

    def mean(self, na_threshold: float = 0.0) -> pd.DataFrame:
        mask = self._data > na_threshold
        data = self._data.copy()
        data[~mask] = np.nan
        mean = data.sum(axis=1) / mask.sum(axis=1)
        return pd.DataFrame({f'mean_{self.name}': mean})

    def filter(self: Type[T],
               min_frequency: Optional[int] = None,
               na_threshold: float = 0.0) -> T:
        raise NotImplementedError

    def frequency(self, na_threshold: float = 0.0) -> pd.DataFrame:
        f = np.sum(self._data > na_threshold, axis=1)
        return pd.DataFrame({f'frequency_{self.name}': f})

    def drop(self: Type[T], exp: Union[str, list], omit_missing_cols: bool = True) -> T:
        if isinstance(exp, str):
            exp = [exp]

        if omit_missing_cols:
            # allow the user to pass column names that don't exist or
            # are already excluded from previous steps.
            exp = [e for e in exp if e in self._data.columns]

        self._data = self._data.drop(exp, axis=1)
        return self

    def _calc_mean_std(self) -> float:
        """
        Calculate the average standard deviation between repeats
        so that you can use it to add random noise during missing value imputation.
        """
        return self._data.std(axis=1, skipna=True).dropna().mean()

    def impute(self: Type[T],
               method: ConditionImputeMethod,
               na_threshold: float = 0.0,
               value: Optional[Union[float, pd.Series]] = None,
               shift: float = 0.0,
               random_noise: bool = False) -> T:
        """
        TBD ...

        Parameters
        ----------
        method
        value
        na_threshold
        shift
        random_noise: bool

        Returns
        -------

        """
        self._data[self._data <= na_threshold] = np.nan

        if random_noise:
            rand_noise_std = self._calc_mean_std()
        else:
            rand_noise_std = None

        if method == 'fixed':
            if value is None:
                raise ValueError(
                    f"To impute missing values with a fixed value,"
                    f" you also need to specify the target fixed value,"
                    f" using the ``value`` argument. Received ``{value}``.")
            self._data = self._data.apply(
                lambda row: self._fillna(row, value, std_value=rand_noise_std),
                axis=1)
        elif method == 'fixed row':
            if value is None:
                raise ValueError(
                    f"To impute missing values with a fixed row value,"
                    f" you also need to specify the target fixed value array,"
                    f" using the ``value`` argument. Received ``{value}``.")
            self._data = self._data.apply(
                lambda row: self._fillna(row, value, std_value=rand_noise_std),
                axis=1)
        elif method == 'row min':
            impute_values = self._data.min(axis=1) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values, std_value=rand_noise_std),
                axis=1)
        elif method == 'row median':
            impute_values = self._data.median(axis=1, skipna=True) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values, std_value=rand_noise_std),
                axis=1)
        elif method == 'row mean':
            impute_values = self._data.mean(axis=1, skipna=True) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values, std_value=rand_noise_std),
                axis=1)
        else:
            raise ValueError(f"Method {method} not implemented")

        return self

    @staticmethod
    def _fillna(
            row: pd.Series,
            val: Union[pd.Series, float],
            std_value: Optional[float] = None) -> pd.Series:
        """
        Fill nan values of a pandas data frame row by row.

        You can either use a fixed value per row, by providing a
        ``pd.Series`` or the same value for all rows, by providing
         a ``float``.  For the first case, the index of the target
        value in the values array, matches the row.name attribute
        of the row.

        If ``std_value`` is specified, Imputed values will be
        selected from a normal distribution with mean the
        value and std the std_value.
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
        Returns the individual experiments from this condition
        as a pandas data frame.

        Returns
        -------
        pd.DataFrame
            A table with protein ids as rows and experiment quantitative
            values as columns.
        """
        return self._data


class TabularDataset(abc.ABC):
    def __init__(self, conditions: List[TabularDatasetExpCondition]) -> None:
        self._conditions = conditions

    @classmethod
    def from_df(cls: Type[T],
                data: pd.DataFrame,
                id_col: str,
                conditions: dict[str, list]) -> T:
        raise NotImplementedError

    @property
    def n_conditions(self) -> int:
        """
        Return the number of experimental conditions included in the dataset.
        """
        return len(self._conditions)

    @property
    def conditions(self):
        """
        Get a list of experimental condition names.

        Returns
        -------
        list
            A list of experimental condition names.
        """
        return [c.name for c in self._conditions]

    @property
    def n_experiments(self) -> int:
        """
        Returns the number of experiment included in the dataset,
        across all experimental conditions.

        Returns
        -------
        int
            Number of experiment included in the dataset.
        """
        n_exp = 0
        for condition in self._conditions:
            n_exp += condition.n_experiments
        return n_exp

    def experiments(self, condition: Optional[str] = None) -> list:
        """
        Get experiment names from the dataset. If experimental condition
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
                exp_names.extend(exp_condition.experiments)

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
        Returns a list of unique protein ids across
        all experimental conditions.
        """
        return self._get_unique_records()

    def _get_unique_records(self) -> List[str]:
        all_records = []
        for condition in self._conditions:
            all_records.extend(condition.record_ids)
        return sorted(list(set(all_records)))

    def describe(self):
        """
        Returns basic information about the dataset.

        Returns
        -------
        dict
            Dataset statistics.
        """
        return {
            'n_conditions_total': self.n_conditions,
            'n_records_total': self.n_records,
            'n_experiments_total': self.n_experiments,
            'statistics_per_condition': [c.describe() for c in self._conditions]
        }

    def mean(self,
             na_threshold: float = 0.0,
             join_method: MergeHow = 'inner') -> pd.DataFrame:
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
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.
        join_method: MergeHow, optional
            Method of joining records of each experimental
            condition in the output.

        Returns
        -------
        pd.DataFrame
            A pandas data frame containing the average value for
            each condition.
        """
        tables = [c.mean(na_threshold=na_threshold) for c in self._conditions]
        return self._join_list_of_tables(tables, how=join_method)

    @staticmethod
    def _join_list_of_tables(tables: List[pd.DataFrame], how: MergeHow = 'outer') -> pd.DataFrame:
        return reduce(lambda left, right: pd.merge(
            left, right, left_index=True,
            right_index=True, how=how), tables)

    def log2_transform(self):
        raise NotImplementedError

    def log2_backtransform(self):
        raise NotImplementedError
