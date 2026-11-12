"""Concatenate multiple objects into one."""
from __future__ import annotations

import copy
from typing import TypeVar

from omicspylib.datasets.peptides import PeptidesDataset
from omicspylib.datasets.proteins import ProteinsDataset

T = TypeVar("T", ProteinsDataset, PeptidesDataset)

def concat(obj: list[T]) -> T:
    """
    Concatenate experimental conditions from multiple datasets, into one dataset.

    Use it when the conditions of an experiment are loaded separately, e.g.
    from different files. Records are matched by their id: a record missing
    from some conditions has missing values there.

    None of the provided datasets is modified.

    Parameters
    ----------
    obj : list[T]
        A list of objects to be concatenated. They should all
        be of the same type and share the same index column names.
        Experimental condition names should not repeat across them.

    Returns
    -------
    T
        A new dataset with the experimental conditions of all
        provided datasets, in the order they were given.

    Raises
    ------
    ValueError
        If ``obj`` is empty, the datasets are of different types or use
        different id column names, or an experimental condition name repeats.

    See Also
    --------
    omicspylib.datasets.proteins.ProteinsDataset.append : Combine two datasets.

    Examples
    --------
    >>> import pandas as pd
    >>> from omicspylib import ProteinsDataset, concat
    >>> control = ProteinsDataset.from_df(
    ...     pd.DataFrame({"protein_id": ["P02768", "P01024"],
    ...                   "ctrl_rep1": [25010.3, 1310.8], "ctrl_rep2": [24410.0, 1450.2]}),
    ...     id_col="protein_id",
    ...     conditions={"control": ["ctrl_rep1", "ctrl_rep2"]},
    ... )
    >>> treated = ProteinsDataset.from_df(
    ...     pd.DataFrame({"protein_id": ["P02768", "P0DJI8"],
    ...                   "trt_rep1": [31200.5, 395.1], "trt_rep2": [30988.1, 402.9]}),
    ...     id_col="protein_id",
    ...     conditions={"treated": ["trt_rep1", "trt_rep2"]},
    ... )
    >>> dataset = concat([control, treated])
    >>> dataset.condition_names
    ['control', 'treated']
    >>> dataset.to_table()  # doctest: +NORMALIZE_WHITESPACE
                ctrl_rep1  ctrl_rep2  trt_rep1  trt_rep2
    protein_id
    P01024         1310.8     1450.2       NaN       NaN
    P02768        25010.3    24410.0   31200.5   30988.1
    P0DJI8            NaN        NaN     395.1     402.9
    """
    if not obj:
        raise ValueError("Provide at least one dataset to concatenate.")

    # Modifying our own copy in place avoids copying the growing dataset
    # again on every step.
    dset = copy.deepcopy(obj[0])
    for o in obj[1:]:
        dset.append(o, inplace=True)

    return dset
