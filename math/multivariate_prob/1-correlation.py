#!/usr/bin/env python3
"""Module that provides a function to calculate a correlation matrix."""

import numpy as np


def correlation(C):
    """Calculate a correlation matrix from a covariance matrix.

    Args:
        C (numpy.ndarray): A 2D square covariance matrix.

    Raises:
        TypeError: If C is not a numpy.ndarray.
        ValueError: If C is not a 2D square matrix.

    Returns:
        numpy.ndarray: The correlation matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlation = C / outer_v
    correlation[C == 0] = 0
    return correlation
