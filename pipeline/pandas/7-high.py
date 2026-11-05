#!/usr/bin/env python3
"""Module that sorts a pandas.DataFrame by the High column."""


def high(df):
    """Sort a DataFrame by the High column in descending order.

    Args:
        df (pandas.DataFrame): The input DataFrame containing a 'High'
            column.

    Returns:
        pandas.DataFrame: The DataFrame sorted by 'High' values in descending
        order.
    """
    return df.sort_values(by='High', ascending=False)
