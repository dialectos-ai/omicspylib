"""Utility functions for using with MaxQuant output data."""

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


def mq_rm_reverse(data: pd.DataFrame) -> pd.DataFrame:
    """Remove reverse hits. Reverse hits can be either under Reverse or Decoy column,
    depending on the version of MaxQuant used. Both cases are considered here.

    Parameters
    ----------
    data: pd.DataFrame
        A tabular dataset as Pandas data frame.

    Returns
    -------
    pd.DataFrame
        The provided dataset without the reverse/decoy hits, if applicable.
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
            "Both 'Reverse' and 'Decoy' columns are found. Please consider manual filtering of the dataset"
        )
        return data
    else:
        warnings.warn(
            "Neither 'Reverse' nor 'Decoy' columns are found. Please consider manual filtering of the dataset"
        )
        return data


def mq_rm_contaminants(data: pd.DataFrame) -> pd.DataFrame:
    """Remove potential contaminants."""
    filtered_data = data.loc[data["Potential contaminant"] != "+", :].copy()
    logger.info("Removed %d contaminant entries.", len(data) - len(filtered_data))
    return filtered_data


def mq_rm_only_modified(data: pd.DataFrame) -> pd.DataFrame:
    """Remove proteins identified only by modified peptides."""
    filtered_data = data.loc[data["Only identified by site"] != "+", :].copy()
    logger.info(
        "Removed %d 'only identified by site' entries.", len(data) - len(filtered_data)
    )
    return filtered_data
