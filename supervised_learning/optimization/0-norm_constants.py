#!/usr/bin/env python3
"""Module to calculate normalization constants for a dataset."""

import numpy as np


def normalization_constants(X):
    """Calculate the mean and standard deviation of each feature in a dataset.

    Args:
        X (numpy.ndarray): The dataset of shape (m, nx) to be normalized.
            m is the number of data points.
            nx is the number of features.

    Returns:
        tuple: (mean, std)
            mean (numpy.ndarray): The mean of each feature.
            std (numpy.ndarray): The standard deviation of each feature.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
