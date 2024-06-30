from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import pandas as pd

from omicspylib import ProteinsDataset
from omicspylib.datasets.abc import TabularExperimentalConditionDataset, TabularDataset


class PeptidesDatasetExpCondition(TabularExperimentalConditionDataset):
    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 id_col: str,
                 experiment_cols: list,
                 protein_id_col: Optional[str] = None,
                 metadata: Optional[dict] = None) -> None:
        super().__init__(name=name, data=data, id_col=id_col, experiment_cols=experiment_cols)
        # todo - specify in the documentation that id_col should have values - and validate inputs
        # todo - this in not a clean implementation of initializing the object and passing metadata - think of another solution
        self._protein_id_col = protein_id_col
        if metadata is None:
            self._metadata = {
                'peptide_to_protein': {}
            }
            if protein_id_col is not None:
                records = data[[id_col, protein_id_col]].to_dict(orient='records')
                for rec in records:
                    self._metadata['peptide_to_protein'][rec[id_col]] = rec[protein_id_col]
        else:
            self._metadata = copy.deepcopy(metadata)

    def filter(self,
               min_frequency: Optional[int] = None,
               na_threshold: float = 0.0) -> PeptidesDatasetExpCondition:
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

        return PeptidesDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist(),
            protein_id_col=self._protein_id_col,
            metadata=self._metadata)


class PeptidesDataset(TabularDataset):
    @classmethod
    def from_df(cls,
                data: pd.DataFrame,
                id_col: str,
                conditions: dict[str, list],
                protein_id_col: Optional[str] = None) -> PeptidesDataset:
        """
        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing peptide data.
        id_col : str
            The name of the column in the DataFrame that represents the peptide IDs.
            These ids should be unique across rows.
        conditions : dict[str, list]
            A dictionary mapping condition names to lists of column names
            representing the corresponding experimental conditions in the DataFrame.
        protein_id_col: str, optional
            If specified a will be used to link specific peptides with proteins.

        Returns
        -------
        PeptidesDataset
            A `PeptidesDataset` object created from the input DataFrame.
        """
        exp_conditions = []
        for condition_name, condition_experiments in conditions.items():
            exp_condition_dataset = PeptidesDatasetExpCondition(
                name=condition_name,
                data=data.copy(),
                id_col=id_col,
                experiment_cols=condition_experiments,
                protein_id_col=protein_id_col)
            exp_conditions.append(exp_condition_dataset)
        return cls(conditions=exp_conditions)

    def to_proteins(self) -> ProteinsDataset:
        raise NotImplementedError
