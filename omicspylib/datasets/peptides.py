from __future__ import annotations

import copy
import warnings
from typing import Literal, cast

import pandas as pd

from omicspylib.datasets.abc import TabularDataset, TabularExperimentalConditionDataset
from omicspylib.datasets.proteins import ProteinsDataset

ProteinsAggMethod = Literal["sum", "counts"]


class PeptidesDatasetExpCondition(TabularExperimentalConditionDataset):
    """
    The peptide values of a single experimental condition.

    It holds one experimental condition, such as ``control``, with all its
    experiments (e.g., biological replicates): one row per peptide and one
    column per experiment. If a ``protein_id_col`` is given, it also keeps
    the protein of each peptide, in ``metadata["peptide_to_protein"]``.

    A :class:`~omicspylib.datasets.peptides.PeptidesDataset` is made of one
    of these objects per condition. They are created for you by
    :meth:`PeptidesDataset.from_df
    <omicspylib.datasets.peptides.PeptidesDataset.from_df>`, and the dataset
    methods call them internally, condition by condition. You normally
    don't need to use them directly.

    Unlike the dataset objects, some methods of a condition object modify it
    **in place**: :meth:`impute`, :meth:`shift`, :meth:`log2_transform` and
    :meth:`log2_backtransform`. :meth:`filter` and :meth:`drop` return a new
    object.

    Examples
    --------
    >>> import pandas as pd
    >>> from omicspylib.datasets.peptides import PeptidesDatasetExpCondition
    >>> data = pd.DataFrame({
    ...     "peptide": ["LVNEVTEFAK", "AEFAEVSK", "VGYVSGWGR"],
    ...     "protein_id": ["P02768", "P02768", "P00738"],
    ...     "ctrl_rep1": [1200.0, 830.5, 410.2],
    ...     "ctrl_rep2": [1150.4, 0.0, 398.7],
    ... })
    >>> control = PeptidesDatasetExpCondition(
    ...     name="control",
    ...     data=data,
    ...     id_col="peptide",
    ...     experiment_cols=["ctrl_rep1", "ctrl_rep2"],
    ...     protein_id_col="protein_id",
    ... )
    >>> control.metadata["peptide_to_protein"]["VGYVSGWGR"]
    'P00738'
    >>> control.filter(protein_ids=["P02768"]).record_ids
    ['LVNEVTEFAK', 'AEFAEVSK']
    """

    def __init__(
        self,
        name: str,
        data: pd.DataFrame,
        id_col: str,
        experiment_cols: list,
        protein_id_col: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Create an experimental condition from a pandas data frame.

        Parameters
        ----------
        name: str
            Name of the experimental condition (e.g., ``control``).
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
            If it is not provided, there is no information about doing that conversion.
        metadata: dict, optional
            Used internally to pass the peptide-to-protein mapping to new
            objects created by :meth:`filter` and :meth:`drop`. If not
            provided, the mapping is built from ``protein_id_col``.
        """
        super().__init__(
            name=name, data=data, id_col=id_col, experiment_cols=experiment_cols
        )
        # todo - this in not a clean implementation of initializing the object and
        #  passing metadata - think of another solution
        self._protein_id_col = protein_id_col
        if metadata is None:
            self._metadata = {"peptide_to_protein": {}}
            if protein_id_col is not None:
                records = data[[id_col, protein_id_col]].to_dict(orient="records")
                for rec in records:
                    self._metadata["peptide_to_protein"][rec[id_col]] = rec[
                        protein_id_col
                    ]
        else:
            self._metadata = copy.deepcopy(metadata)

    @property
    def protein_id_col(self) -> str | None:
        """Returns the column name of the protein id column, if available."""
        return self._protein_id_col

    def filter(
        self,
        exp: str | list | None = None,
        min_frequency: int | None = None,
        na_threshold: float = 0.0,
        ids: list | None = None,
        protein_ids: list | None = None,
    ) -> PeptidesDatasetExpCondition:
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
        protein_ids: list, optional
            If specified, keep only the peptides of these proteins.
            Requires ``protein_id_col``.

        Returns
        -------
        PeptidesDatasetExpCondition
            A new object, with the selected experiments and records.
        """
        data = self._apply_filter(
            exp, min_frequency, na_threshold, ids=ids, protein_ids=protein_ids
        )

        return PeptidesDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist(),
            protein_id_col=self._protein_id_col,
            metadata=self._metadata,
        )

    def _apply_filter(
        self, exp, min_frequency, na_threshold, ids=None, protein_ids=None
    ) -> pd.DataFrame:
        data = self._data.copy()
        if min_frequency is not None:
            valid_rows = (data > na_threshold).sum(axis=1) >= min_frequency
            data = data.loc[valid_rows, :].copy()
        if isinstance(exp, str):
            exp = [exp]
        if exp is not None:
            # You might filter by providing names across experimental
            # conditions. So when you work on each condition separately,
            # some names are no longer valid.
            local_exp: list[str] = [ex for ex in exp if ex in data.columns]
            data = data.loc[:, local_exp].copy()
        if ids is not None:
            data = data.loc[data.index.isin(ids)].copy()
        if protein_ids is not None:
            data = self._drop_or_filter_on_protein_ids(
                data, protein_ids, action="filter"
            )

        return cast(pd.DataFrame, data)

    def drop(
        self,
        exp: str | list | None = None,
        ids: list | None = None,
        omit_missing_cols: bool = True,
        **kwargs,
    ) -> PeptidesDatasetExpCondition:
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
        **kwargs
            ``protein_ids`` (list, optional): remove all the peptides of
            these proteins. Requires ``protein_id_col``.

        Returns
        -------
        PeptidesDatasetExpCondition
            A new object, without the specified experiments and records.
        """
        protein_ids: list | None = kwargs.get("protein_ids")

        data = self._apply_drop(
            exp=exp,
            ids=ids,
            protein_ids=protein_ids,
            omit_missing_cols=omit_missing_cols,
        )

        return PeptidesDatasetExpCondition(
            name=self.name,
            data=data.reset_index(),
            id_col=self._id_col,
            experiment_cols=data.columns.tolist(),
            protein_id_col=self._protein_id_col,
            metadata=self._metadata,
        )

    def _apply_drop(
        self,
        exp: str | list | None = None,
        ids: list | None = None,
        omit_missing_cols: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        protein_ids = kwargs.get("protein_ids")
        data = cast(pd.DataFrame, self._data.copy())
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
            data = self._drop_or_filter_on_protein_ids(data, protein_ids, action="drop")

        return data

    def _drop_or_filter_on_protein_ids(
        self, data, protein_ids, action: Literal["drop", "filter"]
    ) -> pd.DataFrame:
        assert action in ["drop", "filter"]

        # step 1 - make the reverse lookup but save peptide ids in an array
        prot2pept = {}
        for pept_id, prot_id in self._metadata["peptide_to_protein"].items():
            if action == "drop":
                condition = prot_id not in protein_ids
            else:
                condition = prot_id in protein_ids

            if condition:
                if prot_id not in prot2pept:
                    prot2pept[prot_id] = []
                prot2pept[prot_id].append(pept_id)

        # step 2 - collect target peptide ids
        targ_pept_ids = []
        for _, pept_ids in prot2pept.items():
            targ_pept_ids.extend(pept_ids)

        # step 3 - do the filtering
        return data.loc[data.index.isin(targ_pept_ids)].copy()


class PeptidesDataset(TabularDataset):
    """
    Quantitative peptide data, from one or more experimental conditions.

    The dataset is a table with one row per peptide (record) and one column
    per experiment, such as a biological replicate. Experiments are grouped
    into experimental conditions, such as ``control`` and ``treated``, which
    you define when you create the dataset.

    It has the same interface as
    :class:`~omicspylib.datasets.proteins.ProteinsDataset`. In addition, if you
    provide the protein each peptide belongs to (``protein_id_col`` in
    :meth:`from_df`), you can:

    * aggregate peptides into a ``ProteinsDataset`` with :meth:`to_proteins`,
    * keep or remove all peptides of given proteins, with the ``protein_ids``
      argument of :meth:`filter` and :meth:`drop`.

    Each peptide is assumed to belong to a single protein (group).

    Methods that change the data, such as :meth:`filter`, :meth:`impute`,
    :meth:`normalize` or :meth:`log2_transform`, return a new dataset and
    leave the original unchanged, so they can be chained.

    Missing values can be stored as ``0`` or ``NaN``. Most methods also take
    an ``na_threshold`` argument: values equal to or below it are treated as
    missing too.

    See Also
    --------
    omicspylib.datasets.proteins.ProteinsDataset : The same interface, for
        protein-level data.
    omicspylib.datasets.concat.concat : Combine datasets with different
        experimental conditions.

    Examples
    --------
    Create a dataset of four peptides from two proteins:

    >>> import pandas as pd
    >>> from omicspylib import PeptidesDataset
    >>> data = pd.DataFrame({
    ...     "peptide": ["LVNEVTEFAK", "AEFAEVSK", "QNCELFEQLGEYK", "VGYVSGWGR"],
    ...     "protein_id": ["P02768", "P02768", "P02768", "P00738"],
    ...     "ctrl_rep1": [1200.0, 830.5, 0.0, 410.2],
    ...     "ctrl_rep2": [1150.4, 790.1, 220.3, 398.7],
    ...     "trt_rep1": [1420.8, 905.6, 260.1, 0.0],
    ...     "trt_rep2": [1388.2, 950.0, 241.7, 0.0],
    ... })
    >>> peptides = PeptidesDataset.from_df(
    ...     data,
    ...     id_col="peptide",
    ...     protein_id_col="protein_id",
    ...     conditions={
    ...         "control": ["ctrl_rep1", "ctrl_rep2"],
    ...         "treated": ["trt_rep1", "trt_rep2"],
    ...     },
    ... )
    >>> peptides.n_records
    4

    Sum peptide intensities into protein intensities:

    >>> proteins = peptides.to_proteins(agg_method="sum")
    >>> proteins.to_table()  # doctest: +NORMALIZE_WHITESPACE
                ctrl_rep1  ctrl_rep2  trt_rep1  trt_rep2
    protein_id
    P00738          410.2      398.7       0.0       0.0
    P02768         2030.5     2160.8    2586.5    2579.9

    Keep only the peptides of a protein:

    >>> peptides.filter(protein_ids=["P00738"]).n_records
    1
    """

    _conditions: list[PeptidesDatasetExpCondition]

    @property
    def protein_id_col(self) -> str | None:
        """
        Name of the column that links each peptide to its protein, as given
        to :meth:`from_df`, or ``None`` if it was not provided. It is required
        by :meth:`to_proteins` and by the ``protein_ids`` argument of
        :meth:`filter` and :meth:`drop`.
        """
        first_condition = self._conditions[0]
        return first_condition.protein_id_col

    @classmethod
    def from_df(
        cls,
        data: pd.DataFrame,
        id_col: str,
        conditions: dict[str, list],
        protein_id_col: str | None = None,
    ) -> PeptidesDataset:
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
                protein_id_col=protein_id_col,
            )
            exp_conditions.append(exp_condition_dataset)
        return cls(conditions=exp_conditions)

    def to_proteins(
        self,
        agg_method: ProteinsAggMethod = "sum",
        names_lookup: dict | None = None,
        add_prefix: str | None = None,
    ) -> ProteinsDataset | None:
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

                * ``sum``: e.g., calculate protein intensity as the sum
                  of individual peptide intensities.
                * ``counts``: e.g., calculate peptide counts per protein.
        names_lookup: dict, optional
            A lookup dictionary used for renaming the column names of the returned dataset.
            Keys of the dictionary are the original column names and values the new ones.
            For simpler cases where you prefix a tag, use ``add_prefix`` instead.
            Note that the names should match exactly.
        add_prefix: str, optional
            Use ``add_prefix`` instead of ``names_lookup`` for simple name prefixing.
            NOTE: there will be no separator between the prefix and the existing column name.
            You need to provide it (e.g., ``intensity_`` or ``pept_counts_``).

        Returns
        -------
        ProteinsDataset
            A :class:`~omicspylib.datasets.proteins.ProteinsDataset`
            derived from the specific instance.
        """
        if self.protein_id_col is None:
            warnings.warn(
                "No protein ID column is declared during the Peptides dataset initialization."
                "Do not have the necessary information to aggregate into proteins. "
                "Returning None.",
                UserWarning,
                stacklevel=2,
            )
            return None

        assert agg_method in ["sum", "counts"]
        cond_conf = {}

        # since each experimental condition might have a fraction of the total records
        # loop over conditions and create a peptide-to-proteins lookup dict including
        # all the records of all datasets
        pept2proteins = {}
        for condition in self._conditions:
            record = {condition.name: condition.experiment_names}
            cond_conf.update(record)
            metadata = condition.metadata
            pept2proteins.update(metadata["peptide_to_protein"])

        protein_id_col = self._conditions[0].protein_id_col

        data = self.to_table()
        data[protein_id_col] = [
            pept2proteins.get(i, "<unk>") for i in data.index.tolist()
        ]
        if agg_method == "counts":
            numeric_cols = [c for c in data.columns if c != protein_id_col]
            data[numeric_cols] = (data[numeric_cols] > 0).astype(int)

        aggregate_df = data.groupby(protein_id_col).sum().reset_index()

        if names_lookup is not None:
            # since column names are passed in the conditions argument,
            # update the conditions' configuration accordingly.
            for val in cond_conf.values():
                for i, v in enumerate(val):
                    if v in names_lookup:
                        val[i] = names_lookup[v]
            aggregate_df = aggregate_df.rename(columns=names_lookup)
        elif add_prefix is not None:
            prefixed_names = {}
            for val in cond_conf.values():
                for i, v in enumerate(val):
                    val[i] = add_prefix + v
                    prefixed_names[v] = val[i]
            aggregate_df = aggregate_df.rename(columns=prefixed_names)

        # cond_conf is updated in place to match the new names
        return ProteinsDataset.from_df(
            data=aggregate_df, id_col=self.protein_id_col, conditions=cond_conf
        )

    def append(
        self,
        new_obj: PeptidesDataset,
        skip_duplicates: bool = False,
        inplace: bool | None = None,
    ) -> PeptidesDataset:
        """
        Append the experimental conditions of another dataset to this one.

        ``new_obj`` is never modified. To combine more than two datasets,
        see :func:`~omicspylib.datasets.concat.concat`.

        Parameters
        ----------
        new_obj: PeptidesDataset
            A Peptides dataset object to join.
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
        PeptidesDataset:
            A dataset containing the experimental conditions of the two
            datasets. With ``inplace=True``, it is this same object.

        Raises
        ------
        ValueError:
            If the provided class differs from the existing or the id_col or
            protein_id_col, column name differs, or an experimental condition
            already exists.
        """
        # Only peptide datasets have a protein_id_col. Other types are rejected
        # with a clear error in ``_append_conditions``.
        if self.__class__ == new_obj.__class__:
            protein_id_col = self.protein_id_col
            if protein_id_col != new_obj.protein_id_col:
                raise ValueError(
                    f"Cannot join, because there is a mismatch between the "
                    f"protein_id_col name between the datasets. All datasets should "
                    f"have a protein_id_col == {protein_id_col}."
                )
        return self._append_conditions(new_obj, skip_duplicates=skip_duplicates, inplace=inplace)

    def filter(
        self,
        exp: str | list | None = None,
        cond: list | None = None,
        min_frequency: int | None = None,
        na_threshold: float = 0.0,
        ids: list | None = None,
        protein_ids: list | None = None,
    ) -> PeptidesDataset:
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
                c.filter(
                    exp=exp,
                    min_frequency=min_frequency,
                    na_threshold=na_threshold,
                    ids=ids,
                    protein_ids=protein_ids,
                )
                for c in exp_conditions
            ]

        return self.__class__(conditions=exp_conditions)

    def drop(
        self,
        exp: str | list | None = None,
        cond: str | list | None = None,
        ids: list | None = None,
        protein_ids: list | None = None,
    ) -> PeptidesDataset:
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
