from __future__ import annotations
from typing import List, Optional

import pandas as pd

from omicspylib import ProteinsDataset
from omicspylib.datasets.abc import TabularExperimentalConditionDataset, TabularDataset


class PeptidesDatasetExpCondition(TabularExperimentalConditionDataset):
    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 id_col: str,
                 experiment_cols: list,
                 protein_id_col: Optional[str] = None) -> None:
        super().__init__(name=name, data=data, id_col=id_col, experiment_cols=experiment_cols)
        # todo - specify in the documentation that id_col should have values - and validate inputs
        self._metadata = {
            'peptide_to_protein': {}
        }
        if protein_id_col is not None:
            records = data[[id_col, protein_id_col]].to_dict(orient='records')
            for rec in records:
                self._metadata['peptide_to_protein'][rec[id_col]] = rec[protein_id_col]


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
