#!/usr/bin/env python3
"""Module that concatenates and reorganizes DataFrames with multi-index."""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """Concatenate sliced DataFrames and rearrange multi-index hierarchy.

    Args:
        df1 (pandas.DataFrame): The coinbase DataFrame containing a
            'Timestamp' column.
        df2 (pandas.DataFrame): The bitstamp DataFrame containing a
            'Timestamp' column.

    Returns:
        pandas.DataFrame: The concatenated DataFrame filtered between
        timestamps 1417411980 and 1417417980, sorted by multi-index.
    """
    df1 = index(df1)
    df2 = index(df2)
    df = pd.concat(
        [df2.loc[1417411980:1417417980], df1.loc[1417411980:1417417980]],
        keys=['bitstamp', 'coinbase']
    )
    df = df.swaplevel(0, 1).sort_index()
    return df
