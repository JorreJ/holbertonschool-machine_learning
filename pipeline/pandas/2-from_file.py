#!/usr/bin/env python3
"""Module that loads data from a file into a pandas.DataFrame."""

import pandas as pd


def from_file(filename, delimiter):
    """Load data from a file into pandas.DataFrame with a specified delimiter.

    Args:
        filename (str): The path to the file to load.
        delimiter (str): The column separator string.

    Returns:
        pandas.DataFrame: Loaded DataFrame containing the data from the file.
    """
    return pd.read_csv(filename, sep=delimiter)
