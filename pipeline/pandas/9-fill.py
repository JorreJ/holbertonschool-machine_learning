#!/usr/bin/env python3
"""Module that cleans and fills missing values in a pandas.DataFrame."""


def fill(df):
    """Fill missing values in a DataFrame following specific imputation rules.

    Args:
        df (pandas.DataFrame): The input DataFrame containing financial columns
            like 'Close', 'High', 'Low', 'Open', 'Volume_(BTC)',
            'Volume_(Currency)', and 'Weighted_Price'.

    Returns:
        pandas.DataFrame: The modified DataFrame with missing values filled and
        'Weighted_Price' removed.
    """
    df = df.drop(columns=["Weighted_Price"])
    df[["Close"]] = df[["Close"]].ffill()
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    df[["Volume_(BTC)", "Volume_(Currency)"]] = (
        df[["Volume_(BTC)", "Volume_(Currency)"]].fillna(value=0)
    )
    return df
