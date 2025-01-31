from __future__ import annotations

import copy
from typing import Optional, Union, Literal

import numpy as np
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

    @property
    def protein_id_col(self) -> str:
        """Returns the column name of the proteins id column."""
        return self._protein_id_col

    def filter(self,
               exp: Optional[Union[str, list]] = None,
               min_frequency: Optional[int] = None,
               na_threshold: float = 0.0,
               ids: Optional[list] = None,
               protein_ids: Optional[list] = None) -> PeptidesDatasetExpCondition:
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
        ids: list, optional
            If specified, a list of records ids to keep in the dataset.
        protein_ids: list, optional
            If specified, a list of protein ids from which the records will
            remain in the dataset.

        Returns
        -------
        PeptidesDatasetExpCondition
            A new instance of the dataset object, filtered based on the
            user's input.
        """
        data = self._apply_filter(exp, min_frequency, na_threshold, ids=ids, protein_ids=protein_ids)

        return PeptidesDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist(),
            protein_id_col=self._protein_id_col,
            metadata=self._metadata)

    def _apply_filter(self, exp, min_frequency, na_threshold, ids=None, protein_ids=None) -> pd.DataFrame:
        data = self._data.copy()
        if min_frequency is not None:
            valid_rows = np.sum(data > na_threshold, axis=1) >= min_frequency
            data = data.loc[valid_rows, :].copy()
        if isinstance(exp, str):
            exp = [exp]
        if exp is not None:
            # You might filter by providing names across experimental
            # conditions. So when you work on each condition separately,
            # some names are no longer valid.
            local_exp = [ex for ex in exp if ex in data.columns]
            data = data[local_exp].copy()
        if ids is not None:
            data = data.loc[data.index.isin(ids)].copy()
        if protein_ids is not None:
            data = self._drop_or_filter_on_protein_ids(data, protein_ids, action='filter')

        return data

    def drop(self,
             exp: Optional[Union[str, list]] = None,
             ids: Optional[list] = None,
             protein_ids: Optional[list] = None,
             omit_missing_cols: bool = True) -> PeptidesDatasetExpCondition:
        """
        Drop experiments or records from a dataset.

        Parameters
        ----------
        exp: list, str, optional
            List or experiment to drop from the dataset.
            Leave empty to keep all experiments.
        ids: list, optional
            If specified, a list of records ids to keep in the dataset.
        protein_ids: list, optional
            If specified, a list of protein ids from which the records will
            be dropped from the dataset.
        omit_missing_cols: bool, optional
            By default, specified columns that do not exist in the dataset
            are omitted. Set to ``False`` to raise exception instead.

        Returns
        -------
        PeptidesDatasetExpCondition
            A new instance of the dataset object, filtered based on the
            user's input.
        """
        data = self._apply_drop(exp=exp, ids=ids,
                                protein_ids=protein_ids,
                                omit_missing_cols=omit_missing_cols)

        return PeptidesDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist(),
            protein_id_col=self._protein_id_col,
            metadata=self._metadata)

    def _apply_drop(self,
                    exp: Optional[Union[str, list]] = None,
                    ids: Optional[list] = None,
                    protein_ids: Optional[list] = None,
                    omit_missing_cols: bool = True) -> pd.DataFrame:
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

        if protein_ids is not None:
            data = self._drop_or_filter_on_protein_ids(data, protein_ids, action='drop')

        return data

    def _drop_or_filter_on_protein_ids(self, data, protein_ids, action: Literal['drop', 'filter']):
        assert action in ['drop', 'filter']

        # step 1 - make the reverse lookup but save peptide ids in an array
        prot2pept = {}
        for pept_id, prot_id in self._metadata['peptide_to_protein'].items():
            if action == 'drop':
                condition = prot_id not in protein_ids
            else:
                condition = prot_id in protein_ids

            if condition:
                if prot_id not in prot2pept:
                    prot2pept[prot_id] = []
                prot2pept[prot_id].append(pept_id)

        # step 2 - collect target peptide ids
        targ_pept_ids = []
        for prot_id, pept_ids in prot2pept.items():
            targ_pept_ids.extend(pept_ids)

        # step 3 - do the filtering
        return data.loc[data.index.isin(targ_pept_ids)].copy()


class PeptidesDataset(TabularDataset):
    """
    A peptides dataset object.
    It contains multiple experimental conditions with one
    or more experiments per condition.
    """

    @property
    def protein_id_col(self) -> str:
        """Get protein ID column."""
        return self._conditions[0].protein_id_col

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

    def to_proteins(self, agg_method: ProteinsAggMethod = 'sum',
                    names_lookup: dict | None = None,
                    add_prefix: str | None = None) -> ProteinsDataset:
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
        names_lookup: dict, optional
            A lookup dictionary used for renaming the column names of the returned dataset.
            Keys of the dictionary are the original column names and values the new ones.
            For simpler cases where you prefix a tag, use ``add_prefix`` instead.
            Note that the names should match exactly.
        add_prefix: str, optional
            Use ``add_prefix`` instead of ``names_lookup`` for simple name prefixing.
            NOTE: there will be no seperator between prefix and existing column name.
            You need to provide it (e.g., intensity_ or pept_counts_).

        Returns
        -------
        ProteinsDataset
            A :class:`~omicspylib.datasets.proteins.ProteinsDataset`
            derived from the specific instance.
        """
        assert agg_method in ['sum', 'counts']
        cond_conf = {}

        # since each experimental condition might have a fraction of the total records
        # loop over conditions and create a peptide-to-proteins lookup dict including
        # all the records of all datasets
        pept2proteins = {}
        for condition in self._conditions:
            record = {condition.name: condition.experiment_names}
            cond_conf.update(record)
            metadata = condition.metadata
            pept2proteins.update(metadata['peptide_to_protein'])

        protein_id_col = self._conditions[0].protein_id_col

        data = self.to_table()
        data[protein_id_col] = [pept2proteins.get(i, '<unk>') for i in data.index.tolist()]
        if agg_method == 'counts':
            numeric_cols = [c for c in data.columns if c != protein_id_col]
            data[numeric_cols] = (data[numeric_cols] > 0).astype(int)

        aggregate_df = data.groupby(protein_id_col).sum().reset_index()

        if names_lookup is not None:
            # since column names are passed in the conditions argument,
            # update the conditions' configuration accordingly.
            for _, val in cond_conf.items():
                for i, v in enumerate(val):
                    if v in names_lookup:
                        val[i] = names_lookup[v]
            aggregate_df = aggregate_df.rename(columns=names_lookup)
        elif add_prefix is not None:
            prefixed_names = {}
            for _, val in cond_conf.items():
                for i, v in enumerate(val):
                    val[i] = add_prefix + v
                    prefixed_names[v] = val[i]
            aggregate_df = aggregate_df.rename(columns=prefixed_names)

        # cond_conf is updated in place to match the new names
        return ProteinsDataset.from_df(data=aggregate_df, id_col=self.protein_id_col, conditions=cond_conf)

    def append(self, new_obj: PeptidesDataset, skip_duplicates: bool = False) -> PeptidesDataset:
        """
        Append another experimental condition in the same dataset.

        Parameters
        ----------
        new_obj: PeptidesDataset
            A Peptides dataset object to join.
        skip_duplicates: bool
            If ``False``, when an experimental condition (name) already exists,
            it will raise an error.
            Otherwise, it will just be omitted.

        Returns
        -------
        PeptidesDataset:
            A new object containing the experimental conditions of the two datasets.

        Raises
        ------
        ValueError:
            If the provided class differs from the existing or the id_col or
            protein_id_col, column name differs, or an experimental condition
            already exists.
        """
        if not self.__class__ == new_obj.__class__:
            raise ValueError(
                f'The provided object should be of type {self.__class__}. '
                f'Received object of type {new_obj.__class__} instead.f')
        id_col = self._conditions[0].id_col
        protein_id_col = self._conditions[0].protein_id_col

        if not id_col == new_obj._conditions[0].id_col:
            raise ValueError(
                f'Cannot join, because there is a missmatch between the '
                f'id_col name between the datasets. All datasets should '
                f'have an id_col == {id_col}.'
            )
        if not protein_id_col == new_obj._conditions[0].protein_id_col:
            raise ValueError(
                f'Cannot join, because there is a missmatch between the '
                f'protein_id_col name between the datasets. All datasets should '
                f'have a protein_id_col == {protein_id_col}.'
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

    def filter(self,
               exp: Optional[Union[str, list]] = None,
               cond: Optional[list] = None,
               min_frequency: Optional[int] = None,
               na_threshold: float = 0.0,
               ids: Optional[list] = None,
               protein_ids: Optional[list] = None) -> PeptidesDataset:
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
        protein_ids: list, optional
            If specified, a list of protein ids from which the records will
            remain in the dataset.

        Returns
        -------
        ProteinsDataset
            A new instance of the dataset object, filtered based on the
            user's input.
        """
        exp_conditions = self._conditions.copy()
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
                c.filter(exp=exp,
                         min_frequency=min_frequency,
                         na_threshold=na_threshold,
                         ids=ids,
                         protein_ids=protein_ids) for c in exp_conditions]

        return self.__class__(conditions=exp_conditions)

    def drop(self,
             exp: Optional[Union[str, list]] = None,
             cond: Optional[Union[str, list]] = None,
             ids: Optional[list] = None,
             protein_ids: Optional[list] = None) -> PeptidesDataset:
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
        protein_ids: list, optional
            If specified, a list of protein ids from which the records will
            be dropped from the dataset.

        Returns
        -------
        PeptidesDataset
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
        if protein_ids is not None:
            filt_conditions = [c.drop(protein_ids=protein_ids) for c in filt_conditions]

        return self.__class__(conditions=filt_conditions)
