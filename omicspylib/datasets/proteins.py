"""
Proteins dataset object definition.
"""
from __future__ import annotations

import copy
from functools import reduce
from typing import List, Tuple, Optional, Literal, Union

import numpy as np
import pandas as pd

MergeHow = Literal['left', 'right', 'inner', 'outer', 'cross']
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
    'group row median',
    'group row mean with noise'
]

ConditionImputeMethod = Literal[
    'fixed',
    'fixed row',
    'row min',
    'row mean',
    'row median'
    'row mean with noise'
]
AxisName = Literal['rows', 'columns']


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

    def log2_backtransform(self) -> ProteinsDatasetExpCondition:
        self._data = 2**self._data - 1
        return self

    def impute(self,
               method: ConditionImputeMethod,
               na_threshold: float = 0.0,
               value: Optional[Union[float, pd.Series]] = None,
               shift: float = 0.0) -> ProteinsDatasetExpCondition:
        """
        TBD ...

        Parameters
        ----------
        method
        value
        na_threshold
        shift

        Returns
        -------

        """
        self._data[self._data <= na_threshold] = np.nan

        if method == 'fixed':
            if value is None:
                raise ValueError(
                    f"To impute missing values with a fixed value,"
                    f" you also need to specify the target fixed value,"
                    f" using the ``value`` argument. Received ``{value}``.")
            self._data.fillna(value=value, inplace=True)
        elif method == 'fixed row':
            if value is None:
                raise ValueError(
                    f"To impute missing values with a fixed row value,"
                    f" you also need to specify the target fixed value array,"
                    f" using the ``value`` argument. Received ``{value}``.")
            self._data = self._data.apply(
                lambda row: self._fillna(row, value),  # type: ignore
                axis=1)  # type: ignore
        elif method == 'row min':
            impute_values = self._data.min(axis=1) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values),  # type: ignore
                axis=1)
        elif method == 'row median':
            impute_values = self._data.median(axis=1, skipna=True) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values),  # type: ignore
                axis=1)
        elif method == 'row mean':
            impute_values = self._data.mean(axis=1, skipna=True) - shift
            self._data = self._data.apply(
                lambda row: self._fillna(row, impute_values),  # type: ignore
                axis=1)
        elif method == 'row mean with noise':
            mean_values = self._data.mean(axis=1, skipna=True) - shift
            std_values = self._data.std(axis=1, skipna=True)
            self._data = self._data.apply(
                lambda row: self._fillna(row, mean_values, std_values),  # type: ignore
                axis=1)
        else:
            raise ValueError(f"Method {method} not implemented")

        return self

    @staticmethod
    def _fillna(
            row: pd.Series,
            values: pd.Series,
            std_values: Optional[pd.Series] = None) -> pd.Series:
        """
        Fill nan values of a pandas data frame row with the
        specified value in the values array.
        The index of the target value in the values array,
        matches the row.name attribute of the row.
        """
        val_idx = np.where(row.name == values.index)[0][0]
        if std_values is None:
            row = row.fillna(values[val_idx])
        else:
            std_idx = np.where(row.name == std_values.index)[0][0]
            for j, val in enumerate(row):
                if np.isnan(val):
                    row[j] = np.random.normal(values[val_idx], std_values[std_idx])

        return row

    def drop(self, exp: Union[str, list], omit_missing_cols: bool = True) -> ProteinsDatasetExpCondition:
        if isinstance(exp, str):
            exp = [exp]

        if omit_missing_cols:
            # allow the user to pass column names that don't exist or
            # are already excluded from previous steps.
            exp = [e for e in exp if e in self._data.columns]

        self._data = self._data.drop(exp, axis=1)
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
        conditions_copy = copy.deepcopy(self._conditions)
        bt_conditions = [c.log2_backtransform() for c in conditions_copy]
        return ProteinsDataset(bt_conditions)

    def impute(self,
               method: ImputeMethod,
               na_threshold: float = 0.0,
               value: Optional[float] = None,
               shift: float = 0.0) -> ProteinsDataset:
        """
        Impute missing values.

        Parameters
        ----------
        method: str
            Imputation method. Can be one of:
                - fixed: A fixed value. All values below the given threshold
                  will be set to that value. To use this method you also need
                  to specify the `value` parameter.
                - global min: First the min value of the dataset is calculated
                  and then missing values are set to that fixed value. You can
                  also specify the `shift` parameter to shift the calculated min
                  by a fixed step.
                - global row min|median|mean: Similar to ``global min`` but the
                  min|median|mean value refers to the row entry value instead of
                  the value across all entries of that table.
        na_threshold: float, optional
            Values below or equal to this threshold are considered missing.
        value: float, optional
            If ``fixed`` method is specified, you also need to set that value here.
        shift: float, optional
            If ``global|group-min`` method is specified, you can also decrease
            that value by a fixed step.
        """
        imputed_conditions = copy.deepcopy(self._conditions)
        if method == 'fixed':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'fixed', na_threshold, value)
        elif method == 'global min':
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions, 'min', na_threshold, shift)
        elif method == 'global mean':
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions, 'mean', na_threshold, shift)
        elif method == 'global median':
            imputed_conditions = self._impute_by_global_value(
                imputed_conditions, 'median', na_threshold, shift)
        elif method == 'global row min':
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions, 'min', na_threshold, shift)
        elif method == 'global row mean':
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions, 'mean', na_threshold, shift)
        elif method == 'global row median':
            imputed_conditions = self._impute_by_global_row_value(
                imputed_conditions, 'median', na_threshold, shift)
        elif method == 'group row min':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'row min', na_threshold, value, shift)
        elif method == 'group row mean':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'row mean', na_threshold, value, shift)
        elif method == 'group row median':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'row median', na_threshold, value, shift)
        elif method == 'group row mean with noise':
            imputed_conditions = self._intra_group_imputation(
                imputed_conditions, 'row mean with noise', na_threshold, value, shift)

        return ProteinsDataset(imputed_conditions)

    def _impute_by_global_row_value(self, imputed_conditions, value_type, na_threshold, shift):
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
            c.impute(method='fixed row', na_threshold=na_threshold, value=values_series - shift)
            for c in imputed_conditions]
        return imputed_conditions

    def _impute_by_global_value(self, imputed_conditions, value_type, na_threshold, shift):
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
            value=targ_value - shift)

    @staticmethod
    def _intra_group_imputation(
            conditions,
            method,
            na_threshold,
            value,
            shift: float = 0.0) -> List[ProteinsDatasetExpCondition]:
        imputed_conditions = [
            c.impute(method=method, na_threshold=na_threshold, value=value, shift=shift)
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
