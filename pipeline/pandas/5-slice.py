#!/usr/bin/env python3
"""Module that slices and filters specific columns of a pandas.DataFrame."""


def slice(df):
    """Slice a DataFrame to select specific columns and sample every 60th row.

    Args:
        df (pandas.DataFrame): The input DataFrame containing at least 'High',
            'Low', 'Close', and 'Volume_(BTC)' columns.

    Returns:
        pandas.DataFrame: The sliced DataFrame sampled every 60 rows with only
        the selected columns.
    """
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df.iloc[::60]
