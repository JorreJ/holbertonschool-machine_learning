#!/usr/bin/env python3
"""Module to normalize a dataset using mean and standard deviation."""


def normalize(X, m, s):
    """Normalize a specific dataset.

    Args:
        X (numpy.ndarray): The dataset of shape (d1, d2, ...) to be normalized.
        m (numpy.ndarray): The mean of each feature in X.
        s (numpy.ndarray): The standard deviation of each feature in X.

    Returns:
        numpy.ndarray: The normalized dataset.
    """
    return (X - m) / s
