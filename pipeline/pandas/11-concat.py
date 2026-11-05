#!/usr/bin/env python3
"""Module that concatenates two pandas.DataFrames with keys."""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """Concatenate two DataFrames after setting Timestamp as index.

    Args:
        df1 (pandas.DataFrame): The coinbase DataFrame.
        df2 (pandas.DataFrame): The bitstamp DataFrame.

    Returns:
        pandas.DataFrame: The concatenated DataFrame indexed by keys
        'bitstamp' and 'coinbase'.
    """
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat(
        [df2.loc[df2.index <= 1417411920], df1],
        keys=['bitstamp', 'coinbase']
    )
    return df
