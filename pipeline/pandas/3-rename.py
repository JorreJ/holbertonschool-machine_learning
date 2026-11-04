#!/usr/bin/env python3
"""Module that modifies and filters DataFrame columns for datetime mapping."""

import pandas as pd


def rename(df):
    """Rename column Timestamp to Datetime, convert to datetime and filter.

    Args:
        df (pandas.DataFrame): The DataFrame to modify containing at least
            'Timestamp' and 'Close' columns.

    Returns:
        pandas.DataFrame: The modified DataFrame containing only 'Datetime'
        and 'Close' columns.
    """
    df2 = df.rename(columns={'Timestamp': 'Datetime'})
    df2['Datetime'] = pd.to_datetime(df2['Datetime'], unit='s')
    df2 = df2[['Datetime', 'Close']]
    return df2
