#!/usr/bin/env python3
"""Extracts and converts specific DataFrame rows to a numpy array."""


def array(df):
    """Extract the last 10 rows of High and Close columns as a numpy array.

    Args:
        df (pandas.DataFrame): The DataFrame containing at least 'High' and
            'Close' columns.

    Returns:
        numpy.ndarray: 2D array of shape (10, 2) containing the sliced values
        of 'High' and 'Close'.
    """
    df = df[['High', 'Close']]
    df = df.iloc[-10:, :]
    return df.to_numpy()
