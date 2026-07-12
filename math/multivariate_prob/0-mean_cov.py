#!/usr/bin/env python3
"""Module that provides statistics functions for data analysis."""

import numpy as np


def mean_cov(X):
    """Calculate the mean and covariance matrix of a data set.

    Args:
        X (np.ndarray): A 2D array of shape (n, d) containing the data points.

    Raises:
        TypeError: If X is not a 2D numpy.ndarray.
        ValueError: If X contains fewer than 2 data points.

    Returns:
        tuple: A tuple containing:
            - mean (np.ndarray): A 2D array containing the mean of each
              feature.
            - cov (np.ndarray): The covariance matrix of the data set.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    x_center = X - mean
    cov = np.dot(x_center.T, x_center) / (X.shape[0] - 1)
    return mean, cov
