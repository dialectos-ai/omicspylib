"""
Proteins dataset object definition.
"""
from __future__ import annotations

from typing import List

import pandas as pd


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
        Get experimental condition name (e.g. treated, untreated etc).
        """
        return self._name


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
    def exp_conditions(self):
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

    def experiments(self, condition: str | None = None) -> list:
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

    # pylint: disable=too-many-arguments
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
