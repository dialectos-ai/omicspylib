from __future__ import annotations

import copy
from typing import Optional, Union, Literal

import pandas as pd

from omicspylib import ProteinsDataset
from omicspylib.datasets.abc import TabularExperimentalConditionDataset, TabularDataset


ProteinsAggMethod = Literal['sum', 'counts']


class PeptidesDatasetExpCondition(TabularExperimentalConditionDataset):
    """
    Peptide dataset for a specific experimental condition.
    Includes all experiments (runs) for that case.

    Normally, you don't have to interact with this object.
    :class:`~omicspylib.datasets.peptides.PeptidesDataset` wraps multiple
    :class:`~omicspylib.datasets.peptides.PeptidesDatasetExpCondition`
    objects under one group.
    """
    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 id_col: str,
                 experiment_cols: list,
                 protein_id_col: Optional[str] = None,
                 metadata: Optional[dict] = None) -> None:
        """
        Initializes the object.

        Parameters
        ----------
        name: str
            Name of the object.
        data: pd.DataFrame
            Experiments of the specified condition as a Pandas data frame,
            where each column is one experiment.
            This table might contain unrelated columns.
            Only the column names specified under the
            ``id_col`` and ``experiment_cols`` will be used.
        id_col: str
            Column name containing the peptide identifiers.
            It is expected
            that this column is unique.
        experiment_cols: list
            List of the column names for the experiments you want to include
            in this experimental condition.
            All these specified columns
            should be present in the provided data frame.
        protein_id_col: str, optional
            Column name of the protein identifier column (e.g., Uniprot accession number).
            You might need to specify this name to be able to
            convert a :class:`~omicspylib.datasets.peptides.PeptidesDataset`
            to a :class:`~omicspylib.datasets.proteins.ProteinsDataset`.
            If
            it is not provided, there is no information about doing that conversion.
        metadata: dict
            Optional metadata.
        """
        super().__init__(name=name, data=data, id_col=id_col, experiment_cols=experiment_cols)
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
               exp: Optional[Union[str, list]] = None,
               min_frequency: Optional[int] = None,
               na_threshold: float = 0.0) -> PeptidesDatasetExpCondition:
        """
        Filter dataset based on a given set of properties.

        Parameters
        ----------
        exp: list, str, optional
            List or experiment to keep with. Leave empty to keep all experiments.
        min_frequency: int or None, optional
            If specified, records of the dataset will be filtered based on their
            within group frequency.
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

        return PeptidesDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist(),
            protein_id_col=self._protein_id_col,
            metadata=self._metadata)


class PeptidesDataset(TabularDataset):
    """
    A peptides dataset object.
    It contains multiple experimental conditions with one
    or more experiments per condition.
    """
    proteins_id_col = 'protein_id'

    @classmethod
    def from_df(cls,
                data: pd.DataFrame,
                id_col: str,
                conditions: dict[str, list],
                protein_id_col: Optional[str] = None) -> PeptidesDataset:
        """
        Creates a :class:`~omicspylib.datasets.peptides.PeptidesDataset`
        from a Pandas data frame. You might load your data using the
        method of your choice, partially preprocess and then create a dataset
        to abstract missing value imputation, normalization and/or
        statistical analysis between groups.

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
            If specified, will be used to link specific peptides with proteins.

        Returns
        -------
        PeptidesDataset
            A :class:`~omicspylib.datasets.peptides.PeptidesDataset` object
            created from the input DataFrame.
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

    def to_proteins(self, agg_method: ProteinsAggMethod = 'sum', names_lookup: dict | None = None) -> ProteinsDataset:
        """
        Aggregate peptides dataset into Proteins dataset.

        The quantitative value of the returned Proteins dataset depends
        on the aggregation method specified under ``agg_method``.

        It is assumed that each peptide belongs into one protein group.

        A common scenario to use this method is first to
        normalize the peptide intensities and then aggregate to
        protein abundance for further statistical analysis.
        Or from the same peptides dataset, calculate peptide counts
        (``agg_method="counts"``) rename the columns and join with protein
        abundance values.

        Parameters
        ----------
        agg_method: str
            One of ``sum``, ``counts``:

                * ``sum``: e.g., calculate protein intensity as the sum of individual peptide intensities.
                * ``counts``: e.g., calculate peptide counts per protein.
        names_lookup: str
            A lookup dictionary used for renaming the column names of the returned dataset.
            Keys of the dictionary are the original column names and
            values the new ones.
            Note that the names should match exactly.

        Returns
        -------
        ProteinsDataset
            A :class:`~omicspylib.datasets.proteins.ProteinsDataset`
            derived from the specific instance.
        """
        cond_conf = {}

        # since each experimental condition might have a fraction of the total records
        # loop over conditions and create a peptide-to-proteins lookup dict including
        # all the records of all datasets
        pept2proteins = {}
        for condition in self._conditions:
            record = {condition.name: condition.experiments}
            cond_conf.update(record)
            metadata = condition.metadata
            pept2proteins.update(metadata['peptide_to_protein'])

        data = self.to_table()
        data[self.proteins_id_col] = [pept2proteins.get(i, '<unk>') for i in data.index.tolist()]
        if agg_method:
            numeric_cols = [c for c in data.columns if c != self.proteins_id_col]
            data[numeric_cols] = (data[numeric_cols] > 0).astype(int)

        aggregate_df = data.groupby(self.proteins_id_col).sum().reset_index()

        if names_lookup is not None:
            # since column names are passed in the conditions argument,
            # update the conditions' configuration accordingly.
            for _, val in cond_conf.items():
                for i, v in enumerate(val):
                    if v in names_lookup:
                        val[i] = names_lookup[v]
            aggregate_df = aggregate_df.rename(columns=names_lookup)

        return ProteinsDataset.from_df(data=aggregate_df, id_col=self.proteins_id_col, conditions=cond_conf)
