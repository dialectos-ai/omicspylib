"""Concatenate multiple objects into one."""
from __future__ import annotations

import copy
from typing import Union, List

from omicspylib.datasets.peptides import PeptidesDataset
from omicspylib.datasets.proteins import ProteinsDataset


def concat(obj: List[Union[ProteinsDataset, PeptidesDataset]]) -> (
        Union)[ProteinsDataset, PeptidesDataset]:
    """
    Concatenate experimental conditions from multiple datasets, into one dataset.

    Parameters
    ----------
    obj : list[ProteinsDataset | PeptidesDataset]
        A list of objects to be concatenated. They should all
        be of the same type and share the same index column names.

    Returns
    -------
    ProteinsDataset | PeptidesDataset
        A concatenated version of the provided datasets.
    """
    dset = copy.deepcopy(obj[0])
    for o in obj[1:]:
        dset.append(o)

    return dset
