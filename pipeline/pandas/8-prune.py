#!/usr/bin/env python3
"""Module that cleans a pandas.DataFrame by removing missing values."""


def prune(df):
    """Remove rows from a DataFrame containing NaN values in the Close column.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a 'Close'
            column.

    Returns:
        pandas.DataFrame: The pruned DataFrame with missing 'Close' values
        removed.
    """
    return df.dropna(how="any", subset=["Close"])
