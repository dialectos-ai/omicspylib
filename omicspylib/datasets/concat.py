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

    Parameters
    ----------
    obj : list[T]
        A list of objects to be concatenated. They should all
        be of the same type and share the same index column names.

    Returns
    -------
    T
        A concatenated version of the provided datasets.
    """
    dset = copy.deepcopy(obj[0])
    for o in obj[1:]:
        dset.append(o)

    return dset
