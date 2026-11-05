"""
Helpers for cleaning MaxQuant output tables.

MaxQuant marks entries that are usually excluded from the analysis with a
``+`` in dedicated columns: reverse (decoy) hits, potential contaminants,
and proteins identified only by a modification site. Each function removes
one of these groups and returns a new data frame. The number of removed rows
is logged at the ``INFO`` level of the ``omicspylib.utils.mq`` logger.
"""

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


def mq_rm_reverse(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove reverse (decoy) hits.

    Reverse hits are matches to the reversed protein sequences that
    MaxQuant uses to estimate the false discovery rate. They are marked with
    ``+`` in the ``Reverse`` column or, depending on the MaxQuant version, in
    the ``Decoy`` column. Both are supported, and the column name is matched
    regardless of case.

    If neither column is found, or both are, no rows are removed and a
    warning is raised, so that you can filter the data manually.

    Parameters
    ----------
    data: pd.DataFrame
        A MaxQuant output table, such as ``proteinGroups.txt`` or
        ``peptides.txt``.

    Returns
    -------
    pd.DataFrame
        The table without the reverse hits. If no rows could be removed,
        the provided table is returned as is.

    Warns
    -----
    UserWarning
        If neither a ``Reverse`` nor a ``Decoy`` column is found, or if
        both are found.

    Examples
    --------
    >>> import pandas as pd
    >>> from omicspylib.utils import mq_rm_reverse
    >>> data = pd.DataFrame({
    ...     "Protein IDs": ["P02768", "REV__P01024", "P0DJI8"],
    ...     "Reverse": [None, "+", None],
    ... })
    >>> mq_rm_reverse(data)["Protein IDs"].tolist()
    ['P02768', 'P0DJI8']
    """
    reverse_col_candidates = ["reverse", "decoy"]

    col_map = {str(col).lower(): col for col in data.columns}

    found_cols = [col_map[cand] for cand in reverse_col_candidates if cand in col_map]

    if len(found_cols) == 1:
        col = found_cols[0]
        filtered_data = data.loc[data[col] != "+"].copy()
        logger.info("Removed %d reverse/decoy entries.", len(data) - len(filtered_data))
        return filtered_data
    elif len(found_cols) > 1:
        warnings.warn(
            "Both 'Reverse' and 'Decoy' columns were found, so no reverse hits "
            "were removed. Filter the data manually, using the right column.",
            stacklevel=2
        )
        return data
    else:
        warnings.warn(
            "Neither a 'Reverse' nor a 'Decoy' column was found, so no reverse "
            "hits were removed. Check that the data is a MaxQuant output table, "
            "or filter it manually.",
            stacklevel=2
        )
        return data


def mq_rm_contaminants(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove potential contaminants.

    Potential contaminants are proteins commonly found in samples
    regardless of the experiment, such as keratins or trypsin. MaxQuant
    marks them with ``+`` in the ``Potential contaminant`` column.

    Parameters
    ----------
    data: pd.DataFrame
        A MaxQuant output table, such as ``proteinGroups.txt`` or
        ``peptides.txt``.

    Returns
    -------
    pd.DataFrame
        A copy of the table without the potential contaminants.

    Raises
    ------
    KeyError
        If the table has no ``Potential contaminant`` column.

    Examples
    --------
    >>> import pandas as pd
    >>> from omicspylib.utils import mq_rm_contaminants
    >>> data = pd.DataFrame({
    ...     "Protein IDs": ["P02768", "CON__P04264", "P0DJI8"],
    ...     "Potential contaminant": [None, "+", None],
    ... })
    >>> mq_rm_contaminants(data)["Protein IDs"].tolist()
    ['P02768', 'P0DJI8']
    """
    filtered_data = data.loc[data["Potential contaminant"] != "+", :].copy()
    logger.info("Removed %d contaminant entries.", len(data) - len(filtered_data))
    return filtered_data


def mq_rm_only_modified(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove proteins identified only by modified peptides.

    These proteins were identified only through peptides carrying a
    modification site, which makes their identification less reliable.
    MaxQuant marks them with ``+`` in the ``Only identified by site`` column
    of ``proteinGroups.txt``.

    Parameters
    ----------
    data: pd.DataFrame
        A MaxQuant ``proteinGroups.txt`` table.

    Returns
    -------
    pd.DataFrame
        A copy of the table without the proteins identified only by site.

    Raises
    ------
    KeyError
        If the table has no ``Only identified by site`` column.

    Examples
    --------
    >>> import pandas as pd
    >>> from omicspylib.utils import mq_rm_only_modified
    >>> data = pd.DataFrame({
    ...     "Protein IDs": ["P02768", "P01024", "P0DJI8"],
    ...     "Only identified by site": [None, None, "+"],
    ... })
    >>> mq_rm_only_modified(data)["Protein IDs"].tolist()
    ['P02768', 'P01024']
    """
    filtered_data = data.loc[data["Only identified by site"] != "+", :].copy()
    logger.info(
        "Removed %d 'only identified by site' entries.", len(data) - len(filtered_data)
    )
    return filtered_data
