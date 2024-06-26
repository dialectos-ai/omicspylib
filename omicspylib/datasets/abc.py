from __future__ import annotations
import abc
from typing import List, Optional, Union, Literal, Type, TypeVar

import numpy as np
import pandas as pd


AxisName = Literal['rows', 'columns']
T = TypeVar('T', bound='Dataset')


class DatasetExpCondition(abc.ABC):
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


class Dataset(abc.ABC):
    def __init__(self, conditions: List[DatasetExpCondition]) -> None:
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