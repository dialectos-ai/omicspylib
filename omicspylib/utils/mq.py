"""Utility functions for using with MaxQuant output data."""
import pandas as pd


def mq_rm_reverse(data: pd.DataFrame) -> pd.DataFrame:
    """Remove reverse hits."""
    return data.loc[data['Reverse'] != "+", :].copy()


def mq_rm_contaminants(data: pd.DataFrame) -> pd.DataFrame:
    """Remove potential contaminants."""
    return data.loc[data['Potential contaminant'] != "+", :].copy()


def mq_rm_only_modified(data: pd.DataFrame) -> pd.DataFrame:
    """Remove proteins identified only by modified peptides."""
    return data.loc[data['Only identified by site'] != "+", :].copy()
