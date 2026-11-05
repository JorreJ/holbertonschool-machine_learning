#!/usr/bin/env python3
"""Module that computes descriptive statistics for DataFrame columns."""


def analyze(df):
    """Compute descriptive statistics for all columns except Timestamp.

    Args:
        df (pandas.DataFrame): The input DataFrame containing data to analyze
            and optionally a 'Timestamp' column.

    Returns:
        pandas.DataFrame: Summary statistics of the DataFrame excluding the
        'Timestamp' column.
    """
    return df.drop(columns=['Timestamp']).describe()
