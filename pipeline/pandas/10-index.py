#!/usr/bin/env python3
"""Module that sets a specified column as the DataFrame index."""


def index(df):
    """Set the Timestamp column as the index of a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a 'Timestamp'
            column.

    Returns:
        pandas.DataFrame: The modified DataFrame with 'Timestamp' set as index.
    """
    return df.set_index('Timestamp')
