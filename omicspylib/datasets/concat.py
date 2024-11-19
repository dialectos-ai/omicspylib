"""Concatenate multiple objects into one."""
from __future__ import annotations

import copy
from typing import Iterable, Union

from pydantic import RootModel

from omicspylib.datasets.peptides import PeptidesDataset
from omicspylib.datasets.proteins import ProteinsDataset


class ObjTypes(RootModel):
    root: Iterable[Union[ProteinsDataset, PeptidesDataset]]

    class Config:
        arbitrary_types_allowed = True


def concat(obj: ObjTypes) -> Union[ProteinsDataset, PeptidesDataset]:
    """
    Concatenate experimental conditions from multiple datasets, into one dataset.

    Parameters
    ----------
    obj : Iterable[ProteinsDataset | PeptidesDataset]
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
