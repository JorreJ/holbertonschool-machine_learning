#!/usr/bin/env python3
"""Module that converts a numpy.ndarray into a pandas.DataFrame."""

import string
import numpy as np
import pandas as pd


def from_numpy(array):
    """Convert 2D numpy.ndarray into pandas.DataFrame with alphabet headers.

    Args:
        array (numpy.ndarray): 2D array to convert into a DataFrame.

    Returns:
        pandas.DataFrame: The newly created DataFrame with
        uppercase alphabetical column names.
    """
    df = pd.DataFrame(array)
    df.columns = list(string.ascii_uppercase[:df.shape[1]])
    return df
