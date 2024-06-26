"""
Proteins dataset object definition.
"""
from __future__ import annotations

import copy
from typing import List, Tuple, Optional, Literal, Union

import numpy as np
import pandas as pd

from omicspylib.datasets.abc import Dataset, DatasetExpCondition, MergeHow
from omicspylib.utils import mq_rm_contaminants, mq_rm_reverse, mq_rm_only_modified

ImputeMethod = Literal[
    'fixed',
    'global min',
    'global mean',
    'global median',
    'global row min',
    'global row median',
    'global row mean',
    'group row min',
    'group row mean',
    'group row median'
]


class ProteinsDatasetExpCondition(DatasetExpCondition):
    """
    Proteins dataset for a specific experimental condition.
    Includes all experiments (runs) for that case.
    """
    def filter(self,
               min_frequency: Optional[int] = None,
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


class ProteinsDataset(Dataset):
    """
    A proteins dataset object, including multiple experimental
    conditions with one or more experiments per case.
    """
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
            data = mq_rm_contaminants(data)
        if rm_reverse:
            data = mq_rm_reverse(data)
        if rm_only_modified:
            data = mq_rm_only_modified(data)

        return cls.from_df(data, id_col, conditions)

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
        conditions_copy = copy.deepcopy(self._conditions)
        bt_conditions = [c.log2_backtransform() for c in conditions_copy]
        return ProteinsDataset(bt_conditions)

    def impute(self,
               method: ImputeMethod,
               na_threshold: float = 0.0,
               value: Optional[float] = None,
               shift: float = 0.0,
               random_noise: bool = False) -> ProteinsDataset:
        """
        Impute missing values.

        Parameters
        ----------
        method: str
            Imputation method. Can be one of:
                - fixed: A fixed value. All values below the given threshold
                  will be set to that value. To use this method you also need
                  to specify the `value` parameter.
                - global min|mean|median: First the min|mean|median value of
                  the dataset is calculated and then missing values are set
                  to that fixed value. You can also specify the `shift`
                  parameter to shift the calculated min by a fixed step.
                - global row min|mean|median: Similar to ``global min`` but the
                  min|median|mean value refers to the row entry value instead of
                  the value across all entries of that table.
                - `group row min|mean|median`. Similar to the previous but
                  now the min|mean|median is based on the values of the group.
        na_threshold: float, optional
            Values below or equal to this threshold are considered missing.
        value: float, optional
            If ``fixed`` method is specified, you also need to set that value here.
        shift: float, optional
            If ``global|group-min`` method is specified, you can also decrease
            that value by a fixed step.
        random_noise: bool, optional
            If specified random noise based on the global or within group variability
            will be added. Imputed values will be selected from a normal distribution
            with mean the selected value (depending on the method) and std the within
            group or global standard deviation (depending on the method). Because you
            draw random values from a normal distribution, consider transforming your
            data if needed, to approximate it (e.g. apply log2 transformation, if needed).
            After imputation, you can back_transform to the original scale.
        """
        imputed_conditions = copy.deepcopy(self._conditions)
        if method == 'fixed':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'fixed', na_threshold, value,
                random_noise=random_noise)
        elif method == 'global min':
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions, 'min', na_threshold, shift,
                random_noise=random_noise)
        elif method == 'global mean':
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions, 'mean', na_threshold, shift,
                random_noise=random_noise)
        elif method == 'global median':
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions, 'median', na_threshold, shift, random_noise=random_noise)
        elif method == 'global row min':
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions, 'min', na_threshold, shift, random_noise=random_noise)
        elif method == 'global row mean':
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions, 'mean', na_threshold, shift, random_noise=random_noise)
        elif method == 'global row median':
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions, 'median', na_threshold, shift, random_noise=random_noise)
        elif method == 'group row min':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'row min', na_threshold, value, shift, random_noise=random_noise)
        elif method == 'group row mean':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'row mean', na_threshold, value, shift, random_noise=random_noise)
        elif method == 'group row median':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'row median', na_threshold, value, shift, random_noise=random_noise)

        return ProteinsDataset(imputed_conditions)

    def _impute_by_global_row_value(self, imputed_conditions,
                                    value_type, na_threshold, shift, random_noise):
        df = self.to_table()
        df[df <= na_threshold] = np.nan
        if value_type == 'min':
            values_series = df.min(axis=1, skipna=True)
        elif value_type == 'mean':
            values_series = df.mean(axis=1, skipna=True)
        elif value_type == 'median':
            values_series = df.median(axis=1, skipna=True)
        else:
            raise ValueError(f'Value type {value_type} not supported.')

        imputed_conditions = [
            c.impute(method='fixed row',
                     na_threshold=na_threshold,
                     value=values_series - shift,
                     random_noise=random_noise)
            for c in imputed_conditions]
        return imputed_conditions

    def _impute_by_global_value(self, imputed_conditions, value_type,
                                na_threshold, shift, random_noise: bool = False):
        df = self.to_table()
        df[df <= na_threshold] = np.nan
        all_values = df.values.flatten()
        if value_type == 'min':
            targ_value = np.nanmin(all_values)
        elif value_type == 'mean':
            targ_value = np.nanmean(all_values)
        elif value_type == 'median':
            targ_value = np.nanmedian(all_values)
        else:
            raise ValueError(f'Value type {value_type} not supported.')

        return self._intra_group_imputation(
            conditions=imputed_conditions,
            method='fixed',
            na_threshold=na_threshold,
            value=targ_value - shift,
            random_noise=random_noise)

    @staticmethod
    def _intra_group_imputation(
            conditions,
            method,
            na_threshold,
            value,
            shift: float = 0.0,
            random_noise: bool = False) -> List[ProteinsDatasetExpCondition]:
        imputed_conditions = [
            c.impute(method=method, na_threshold=na_threshold, value=value, shift=shift, random_noise=random_noise)
            for c in conditions]
        return imputed_conditions

    def normalize(self) -> ProteinsDataset:
        """Normalize the dataset."""
        raise NotImplementedError

    def filter(self,
               conditions: Optional[list] = None,
               min_frequency: Optional[int] = None,
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
            frequency within the experimental condition.
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

        return ProteinsDataset(conditions=exp_conditions)


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

    def drop(self,
             exp: Optional[Union[str, list]] = None,
             cond: Optional[Union[str, list]] = None) -> ProteinsDataset:
        """
        Drop specified experiment(s) and or condition(s).
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

        return ProteinsDataset(filt_conditions)
