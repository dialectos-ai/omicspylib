"""
Proteins dataset object definition.
"""
from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from omicspylib.datasets.abc import TabularDataset, TabularExperimentalConditionDataset
from omicspylib.utils import mq_rm_contaminants, mq_rm_reverse, mq_rm_only_modified


class ProteinsDatasetExpCondition(TabularExperimentalConditionDataset):
    """
    Proteins dataset for a specific experimental condition.
    Includes all experiments (runs) for that case.

    Normally, you don't have to interact with this object.
    ``ProteinsDataset`` wraps multiple ``ProteinsDatasetExpCondition``
    objects under one group.
    """
    def filter(self,
               exp: Optional[Union[str, list]] = None,
               min_frequency: Optional[int] = None,
               na_threshold: float = 0.0) -> ProteinsDatasetExpCondition:
        """
        Filter dataset based on a given set of properties.

        Parameters
        ----------
        exp: list, str, optional
            List or experiment to keep with. Leave empty to keep all experiments.
        min_frequency: int or None, optional
            If specified, records of the dataset will be filtered to the records with
            greater than or equal the specified frequency.
        na_threshold: float or None, optional
            Values below or equal to this threshold are considered missing.
            It is used in to filter records based on the number of missing values.

        Returns
        -------
        ProteinsDatasetExpCondition
            A new instance of the dataset object, filtered based on the
            user's input.
        """
        data = self._apply_filter(exp, min_frequency, na_threshold)

        return ProteinsDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist())


class ProteinsDataset(TabularDataset):
    """
    A proteins dataset object.
    It contains multiple experimental conditions with one
    or more experiments per condition.
    """
    @classmethod
    def from_df(cls,
                data: pd.DataFrame,
                id_col: str,
                conditions: dict[str, list]) -> ProteinsDataset:
        """
        Initialize a ``ProteinsDataset`` from a pandas dataframe.

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
            exp_condition_dataset = ProteinsDatasetExpCondition(
                name=condition_name,
                data=data.copy(),
                id_col=id_col,
                experiment_cols=condition_experiments)
            exp_conditions.append(exp_condition_dataset)
        return cls(conditions=exp_conditions)

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
            If True, remove contaminant hits from the dataset, by default, True.
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
            data = mq_rm_contaminants(data)
        if rm_reverse:
            data = mq_rm_reverse(data)
        if rm_only_modified:
            data = mq_rm_only_modified(data)

        return cls.from_df(data, id_col, conditions)

    def append(self, new_obj: ProteinsDataset, skip_duplicates: bool = False) -> ProteinsDataset:
        """
        Append another experimental condition in the same dataset.

        Parameters
        ----------
        new_obj: ProteinsDataset
            A Peptides dataset object to join.
        skip_duplicates: bool
            If ``False``, when an experimental condition (name) already exists,
            it will raise an error.
            Otherwise, it will just be omitted.

        Returns
        -------
        ProteinsDataset:
            A new object containing the experimental conditions of the two datasets.

        Raises
        ------
        ValueError:
            If the provided class differs from the existing or the id_col
            column name differs, or an experimental condition already exists.
        """
        if not self.__class__ == new_obj.__class__:
            raise ValueError(
                f'The provided object should be of type {self.__class__}. '
                f'Received object of type {new_obj.__class__} instead.f')
        id_col = self._conditions[0].id_col

        if not id_col == new_obj._conditions[0].id_col:
            raise ValueError(
                f'Cannot join, because there is a missmatch between the '
                f'id_col name between the datasets. All datasets should '
                f'have an id_col == {id_col}.'
            )

        for new_cond in new_obj._conditions:
            if new_cond.name in self.condition_names and not skip_duplicates:
                raise ValueError(
                    f'Experimental condition {new_cond} already exists in '
                    f'the current dataset. Either remove it or select to '
                    f'`skip_duplicates`.'
                )
        self._conditions.extend(new_obj._conditions)

        return self.__class__(conditions=self._conditions)
