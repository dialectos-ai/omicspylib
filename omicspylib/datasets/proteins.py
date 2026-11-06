"""
Proteins dataset object definition.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from omicspylib.datasets.abc import TabularDataset, TabularExperimentalConditionDataset
from omicspylib.utils import mq_rm_contaminants, mq_rm_only_modified, mq_rm_reverse


class ProteinsDatasetExpCondition(TabularExperimentalConditionDataset):
    """
    The protein values of a single experimental condition.

    It holds one experimental condition, such as ``control``, with all its
    experiments (e.g., biological replicates): one row per protein and one
    column per experiment.

    A :class:`~omicspylib.datasets.proteins.ProteinsDataset` is made of one
    of these objects per condition. They are created for you by
    :meth:`ProteinsDataset.from_df
    <omicspylib.datasets.proteins.ProteinsDataset.from_df>`, and the dataset
    methods call them internally, condition by condition. You normally
    don't need to use them directly.

    Unlike the dataset objects, some methods of a condition object modify it
    **in place**: :meth:`impute`, :meth:`shift`, :meth:`log2_transform` and
    :meth:`log2_backtransform`. :meth:`filter` and :meth:`drop` return a new
    object.

    Examples
    --------
    >>> import pandas as pd
    >>> from omicspylib.datasets.proteins import ProteinsDatasetExpCondition
    >>> data = pd.DataFrame({
    ...     "protein_id": ["P02768", "P01024", "P0DJI8"],
    ...     "ctrl_rep1": [25010.3, 1310.8, 0.0],
    ...     "ctrl_rep2": [24410.0, 1450.2, 410.5],
    ... })
    >>> control = ProteinsDatasetExpCondition(
    ...     name="control",
    ...     data=data,
    ...     id_col="protein_id",
    ...     experiment_cols=["ctrl_rep1", "ctrl_rep2"],
    ... )
    >>> control.n_experiments, control.record_ids
    (2, ['P02768', 'P01024', 'P0DJI8'])
    >>> control.min(axis="rows").to_dict()
    {'ctrl_rep1': 1310.8, 'ctrl_rep2': 410.5}
    """

    def filter(
        self,
        exp: str | list | None = None,
        min_frequency: int | None = None,
        na_threshold: float = 0.0,
        ids: list | None = None,
    ) -> ProteinsDatasetExpCondition:
        """
        Keep a subset of the experiments and records.

        Parameters
        ----------
        exp: list, str, optional
            Experiment name, or list of names, to keep. Names that are not
            part of this condition are ignored. Leave empty to keep all
            experiments.
        min_frequency: int or None, optional
            If specified, keep only the records with at least this number of
            values (non-missing) in this condition.
        na_threshold: float or None, optional
            Values below or equal to this threshold are considered missing.
            Used together with ``min_frequency``.
        ids: list, optional
            If specified, a list of record ids to keep.

        Returns
        -------
        ProteinsDatasetExpCondition
            A new object, with the selected experiments and records.
        """
        data = self._apply_filter(exp, min_frequency, na_threshold, ids=ids)

        return ProteinsDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist(),
        )

    def drop(
        self,
        exp: str | list | None = None,
        ids: list | None = None,
        omit_missing_cols: bool = True,
        **kwargs
    ) -> ProteinsDatasetExpCondition:
        """
        Remove experiments or records.

        Parameters
        ----------
        exp: list, str, optional
            Experiment name, or list of names, to remove.
        ids: list, optional
            If specified, a list of record ids to remove.
        omit_missing_cols: bool, optional
            By default, experiment names that are not part of this condition
            are ignored. Set to ``False`` to raise an error instead.

        Returns
        -------
        ProteinsDatasetExpCondition
            A new object, without the specified experiments and records.
        """
        data = self._apply_drop(exp=exp, ids=ids, omit_missing_cols=omit_missing_cols)

        return ProteinsDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist(),
        )


class ProteinsDataset(TabularDataset):
    """
    Quantitative protein data, from one or more experimental conditions.

    The dataset is a table with one row per protein (record) and one column
    per experiment, such as a biological replicate. Experiments are grouped
    into experimental conditions, such as ``control`` and ``treated``, which
    you define when you create the dataset.

    Create a dataset with :meth:`from_df`, from any pandas data frame, or with
    :meth:`from_maxquant`, from a MaxQuant ``proteinGroups.txt`` file.

    Methods that change the data, such as :meth:`filter`, :meth:`impute`,
    :meth:`normalize` or :meth:`log2_transform`, return a new dataset and
    leave the original unchanged, so they can be chained.

    Missing values can be stored as ``0`` or ``NaN``. Most methods also take
    an ``na_threshold`` argument: values equal to or below it are treated as
    missing too.

    See Also
    --------
    omicspylib.datasets.peptides.PeptidesDataset : The same interface, for
        peptide-level data.
    omicspylib.datasets.concat.concat : Combine datasets with different
        experimental conditions.

    Examples
    --------
    Create a dataset with two conditions of three replicates each:

    >>> import pandas as pd
    >>> from omicspylib import ProteinsDataset
    >>> data = pd.DataFrame({
    ...     "protein_id": ["P02768", "P01024", "P0DJI8"],
    ...     "ctrl_rep1": [25010.3, 1310.8, 0.0],
    ...     "ctrl_rep2": [24410.0, 1450.2, 410.5],
    ...     "ctrl_rep3": [26120.7, 1398.7, 0.0],
    ...     "trt_rep1": [31200.5, 0.0, 395.1],
    ...     "trt_rep2": [30988.1, 0.0, 402.9],
    ...     "trt_rep3": [29874.2, 512.4, 388.0],
    ... })
    >>> dataset = ProteinsDataset.from_df(
    ...     data,
    ...     id_col="protein_id",
    ...     conditions={
    ...         "control": ["ctrl_rep1", "ctrl_rep2", "ctrl_rep3"],
    ...         "treated": ["trt_rep1", "trt_rep2", "trt_rep3"],
    ...     },
    ... )
    >>> dataset.condition_names
    ['control', 'treated']
    >>> dataset.n_records, dataset.n_experiments
    (3, 6)

    Count in how many experiments of each condition each protein is found:

    >>> dataset.frequency()  # doctest: +NORMALIZE_WHITESPACE
                frequency_control  frequency_treated
    protein_id
    P01024                      3                  1
    P02768                      3                  3
    P0DJI8                      1                  3

    Keep proteins found in at least 2 experiments of a condition, and
    log-transform the values:

    >>> prepared = dataset.filter(min_frequency=2).log2_transform()
    """

    _conditions: list[ProteinsDatasetExpCondition]

    def __init__(self, conditions: Sequence[ProteinsDatasetExpCondition]) -> None:
        """
        Create a dataset from experimental condition objects.

        You normally don't need to call the constructor directly. Use
        :meth:`from_df` or :meth:`from_maxquant` instead.

        Parameters
        ----------
        conditions: Sequence[ProteinsDatasetExpCondition]
            One object per experimental condition. Each condition name
            should appear only once.
        """
        super().__init__(conditions=list(conditions))
        self._conditions = list(conditions)

    @classmethod
    def from_df(
        cls, data: pd.DataFrame, id_col: str, conditions: dict[str, list]
    ) -> ProteinsDataset:
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
                experiment_cols=condition_experiments,
            )
            exp_conditions.append(exp_condition_dataset)
        return cls(conditions=exp_conditions)

    @classmethod
    def from_maxquant(
        cls,
        data: str | pd.DataFrame,
        conditions: dict[str, list],
        rm_reverse: bool = True,
        rm_contaminants: bool = True,
        rm_only_modified: bool = True,
        id_col: str = "Majority protein IDs",
        rename_id_col: str | None = "protein_id",
    ) -> ProteinsDataset:
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
            data = pd.read_csv(data, sep="\t")

        if rename_id_col is not None:
            data = data.rename(columns={id_col: rename_id_col})
            id_col = rename_id_col

        if rm_contaminants:
            data = mq_rm_contaminants(data)
        if rm_reverse:
            data = mq_rm_reverse(data)
        if rm_only_modified:
            data = mq_rm_only_modified(data)

        return cls.from_df(data, id_col, conditions)

    def append(
        self,
        new_obj: ProteinsDataset,
        skip_duplicates: bool = False,
        inplace: bool | None = None,
    ) -> ProteinsDataset:
        """
        Append the experimental conditions of another dataset to this one.

        ``new_obj`` is never modified. To combine more than two datasets,
        see :func:`~omicspylib.datasets.concat.concat`.

        Parameters
        ----------
        new_obj: ProteinsDataset
            A Proteins dataset object to join.
        skip_duplicates: bool
            If ``False``, when an experimental condition (name) already exists,
            it will raise an error.
            Otherwise, that condition of ``new_obj`` is left out, and the
            existing one is kept.
        inplace: bool, optional
            If ``True``, modify this dataset and return it. If ``False``,
            return a new dataset and leave this one unchanged.
            If not specified, behaves as ``True`` and emits a ``FutureWarning``:
            the default will change to ``False`` in a future release.

        Returns
        -------
        ProteinsDataset:
            A dataset containing the experimental conditions of the two
            datasets. With ``inplace=True``, it is this same object.

        Raises
        ------
        ValueError:
            If the provided class differs from the existing or the id_col
            column name differs, or an experimental condition already exists.
        """
        return self._append_conditions(new_obj, skip_duplicates=skip_duplicates, inplace=inplace)
