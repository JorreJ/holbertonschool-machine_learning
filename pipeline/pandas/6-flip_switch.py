#!/usr/bin/env python3
"""Module that sorts and transposes a pandas.DataFrame."""


def flip_switch(df):
    """Sort a DataFrame by Timestamp in descending order and transpose it.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a 'Timestamp'
            column.

    Returns:
        pandas.DataFrame: The transposed DataFrame after sorting.
    """
    df = df.sort_values(by='Timestamp', ascending=False)
    return df.T
