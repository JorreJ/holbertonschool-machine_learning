#!/usr/bin/env python3
"""Module that converts a numpy.ndarray into a pandas.DataFrame."""

import pandas as pd


def from_numpy(array):
    """Convert 2D numpy.ndarray into pandas.DataFrame with alphabet headers.

    Args:
        array (numpy.ndarray): 2D array to convert into a DataFrame.

    Returns:
        pandas.DataFrame: The newly created DataFrame with uppercase
        alphabetical column names.
    """
    columns = [chr(i) for i in range(ord('A'), ord('A') + array.shape[1])]
    return pd.DataFrame(array, columns=columns)
