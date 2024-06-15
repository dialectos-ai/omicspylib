"""
Proteins dataset object definition.
"""
from __future__ import annotations

import copy
from functools import reduce
from typing import List, Tuple, Union, Literal

import numpy as np
import pandas as pd

MergeHow = Literal["left", "right", "inner", "outer", "cross"]

class ProteinsDatasetExpCondition:
    """
    Proteins dataset for a specific experimental condition.
    Includes all experiments (runs) for that case.
    """
    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 id_col: str,
                 experiment_cols: list) -> None:
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
        self._experiments = experiment_cols

    @property
    def n_experiments(self) -> int:
        """
        Returns the number of experiments.

        Returns
        -------
            int: The number of experiments.
        """
        return len(self._experiments)

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
            'n_proteins_per_experiment': np.sum(self._data.values > 0, axis=0).tolist()
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
        return df, n_missing_total, total_values

    def filter(self,
               min_frequency: Union[int, None] = None,
               na_threshold: float = 0.0) -> ProteinsDatasetExpCondition:
        """
        Filter dataset based on a given set of properties.

        Parameters
        ----------
        min_frequency: int or None, optional
            If specified, records of the dataset will be filtered based on their
            within group frequency.
        na_threshold: float or None, optional
            Values below or equal to this threshold are considered missing.
            Is used in to filter records based on the number of missing values.

        Returns
        -------
        ProteinsDatasetExpCondition
            A new instance of the dataset object, filtered based on the
            user's input.
        """
        data = self._data.copy()
        if min_frequency is not None:
            valid_rows = np.sum(data > na_threshold, axis=1) >= min_frequency
            data = data.loc[valid_rows, :].copy()

        return ProteinsDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist())

    def mean(self, na_threshold: float = 0.0) -> pd.DataFrame:
        mask = self._data > na_threshold
        data = self._data.copy()
        data[~mask] = np.nan
        mean = data.sum(axis=1) / mask.sum(axis=1)
        return pd.DataFrame({f'mean_{self.name}': mean})

    def frequency(self, na_threshold: float = 0.0) -> pd.DataFrame:
        f = np.sum(self._data > na_threshold, axis=1)
        return pd.DataFrame({f'frequency_{self.name}': f})

    def log2_transform(self) -> ProteinsDatasetExpCondition:
        self._data = np.log2(self._data + 1)  # type: ignore
        return self


class ProteinsDataset:
    """
    A proteins dataset object, including multiple experimental
    conditions with one or more experiments per case.
    """
    def __init__(self, exp_conditions: List[ProteinsDatasetExpCondition]) -> None:
        self._conditions = exp_conditions

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

    def experiments(self, condition: Union[str, None] = None) -> list:
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

    @classmethod
    def from_df(cls,
                data: pd.DataFrame,
                id_col: str,
                conditions: dict[str, list]) -> ProteinsDataset:
        """
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing protein data.
        id_col : str
            The name of the column in the DataFrame that represents the protein IDs.
        conditions : dict[str, list]
            A dictionary mapping condition names to lists of column names
            representing the corresponding experimental conditions in the DataFrame.

        Returns
        -------
        ProteinsDataset
            A `ProteinsDataset` object created from the input DataFrame.
        """
        exp_conditions = []
        for condition_name, condition_experiments in conditions.items():
            df = data[[id_col] + condition_experiments].copy()
            exp_condition_dataset = ProteinsDatasetExpCondition(
                name=condition_name,
                data=df,
                id_col=id_col,
                experiment_cols=condition_experiments)
            exp_conditions.append(exp_condition_dataset)
        return cls(exp_conditions=exp_conditions)

    @classmethod
    def from_maxquant(cls, data: str | pd.DataFrame,
                      conditions: dict[str, list],
                      rm_reverse: bool = True,
                      rm_contaminants: bool = True,
                      rm_only_modified: bool = True,
                      id_col: str = 'Majority protein IDs',
                      rename_id_col: str | None = 'protein_id') -> ProteinsDataset:
        """
        Create a ProteinsDataset object from MaxQuant proteinGroups.txt file.

        Parameters
        ----------
        data : str or pd.DataFrame
            The input data as a path to a TSV file or a pandas DataFrame.
        conditions : dict[str, list]
            A dictionary mapping condition names to a list of corresponding samples.
        rm_reverse : bool, optional
            If True, remove reverse hits from the dataset, by default True.
        rm_contaminants : bool, optional
            If True, remove contaminant hits from the dataset, by default True.
        rm_only_modified : bool, optional
            If True, remove proteins with only modified peptides, by default True.
        id_col : str, optional
            The column name containing the protein IDs, by default 'Majority protein IDs'.
        rename_id_col : str or None, optional
            The new column name for the protein IDs after renaming, by default 'protein_id'.

        Returns
        -------
        ProteinsDataset
            The assembled ProteinsDataset object.
        """
        if isinstance(data, str):
            data = pd.read_csv(data, sep='\t')

        if rename_id_col is not None:
            data.rename(columns={id_col: rename_id_col}, inplace=True)
            id_col = rename_id_col

        if rm_contaminants:
            data = cls._rm_contaminants(data)
        if rm_reverse:
            data = cls._rm_reverse(data)
        if rm_only_modified:
            data = cls._rm_only_modified(data)

        return cls.from_df(data, id_col, conditions)

    @staticmethod
    def _rm_reverse(data: pd.DataFrame) -> pd.DataFrame:
        return data.loc[data['Reverse'] != "+", :].copy()

    @staticmethod
    def _rm_contaminants(data: pd.DataFrame) -> pd.DataFrame:
        return data.loc[data['Potential contaminant'] != "+", :].copy()

    @staticmethod
    def _rm_only_modified(data: pd.DataFrame) -> pd.DataFrame:
        return data.loc[data['Only identified by site'] != "+", :].copy()

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

    def to_table(self, join_method: MergeHow = 'outer') -> pd.DataFrame:
        """
        Merge individual experimental conditions to one table.

        Parameters
        ----------
        join_method: MergeHow, optional
            Method of joining records of each experimental condition in the output.

        Returns
        -------
        pd.DataFrame
            A pandas data frame containing all experimental conditions.
        """
        tables = [c.to_table() for c in self._conditions]
        return self._join_list_of_tables(tables, how=join_method)

    @staticmethod
    def _join_list_of_tables(tables: List[pd.DataFrame], how: MergeHow = 'outer') -> pd.DataFrame:
        return reduce(lambda left, right: pd.merge(
            left, right, left_index=True,
            right_index=True, how=how), tables)

    def missing_values(self, na_threshold: float = 0.0) -> (
            Tuple)[pd.DataFrame, int, int]:
        """
        Returns number of missing values per experiment and condition.
        Missing values are considered the cases that are either missing
        or are below the specified threshold.

        Parameters
        ----------
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.

        Returns
        -------
        pd.DataFrame
            A pandas data frame with the number of missing cases per
            experiment and condition.
        int
            Number of missing values.
        int
            Number of values in total
        """
        dfs = []
        n_missing = 0
        n_total = 0
        for cond in self._conditions:
            df, n_missing_cond, n_total_cond = cond.missing_values(na_threshold=na_threshold)
            n_missing += n_missing_cond
            n_total += n_total_cond

            dfs.append(df)
        return pd.concat(dfs), n_missing, n_total

    def log2_transform(self) -> ProteinsDataset:
        """Perform log2 transformation."""
        conditions_copy = copy.deepcopy(self._conditions)
        log2_conditions = [c.log2_transform() for c in conditions_copy]
        return ProteinsDataset(log2_conditions)

    def log2_backtransform(self) -> ProteinsDataset:
        """Invert log2 transformation."""
        raise NotImplementedError

    def impute(self) -> ProteinsDataset:
        """Impute missing values."""
        raise NotImplementedError

    def normalize(self) -> ProteinsDataset:
        """Normalize the dataset."""
        raise NotImplementedError

    def filter(self,
               conditions: Union[list, None] = None,
               min_frequency: Union[int, None] = None,
               na_threshold: float = 0.0) -> ProteinsDataset:
        """
        Filter dataset based on a given set of properties.

        Parameters
        ----------
        conditions: list, optional
            List of experimental condition names. If provided only the conditions
            specified will remain in the dataset.
        min_frequency: int or None, optional
            If specified, records of the dataset will be filtered based on their
            within group frequency.
        na_threshold: float or None, optional
            Values below or equal to this threshold are considered missing.
            Is used in to filter records based on the number of missing values.

        Returns
        -------
        ProteinsDataset
            A new instance of the dataset object, filtered based on the
            user's input.
        """
        exp_conditions = self._conditions.copy()

        if conditions is not None:
            exp_conditions = [c for c in exp_conditions if c.name in conditions]

        if min_frequency:
            exp_conditions = [
                c.filter(min_frequency=min_frequency,
                         na_threshold=na_threshold) for c in exp_conditions]

        return ProteinsDataset(exp_conditions=exp_conditions)

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

    def frequency(self,
                  na_threshold: float = 0.0,
                  join_method: MergeHow = 'outer') -> pd.DataFrame:
        """
        Calculate the number of experiments within each experimental condition
        with quantitative value above the specified threshold,
        and return a merged data frame for all conditions.

        By default, and outer join is performed across all conditions.
        Adjust accordingly if needed.

        Parameters
        ----------
        na_threshold : float, optional
            Values below or equal to this threshold are considered missing.
        join_method: MergeHow, optional
            Method of joining records of each experimental condition in the output.

        Returns
        -------
        pd.DataFrame
            A pandas data frame containing the average value for each condition.
        """
        tables = [c.frequency(na_threshold=na_threshold) for c in self._conditions]
        return self._join_list_of_tables(tables, how=join_method)
