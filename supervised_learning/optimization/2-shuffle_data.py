#!/usr/bin/env python3
"""Module to shuffle two matrices synchronized by their first dimension."""

import numpy as np


def shuffle_data(X, Y):
    """Shuffle the data points in two matrices in the same way.

    Args:
        X (numpy.ndarray): The first matrix of shape (m, nx) to shuffle.
            m is the number of data points.
            nx is the number of features.
        Y (numpy.ndarray): The second matrix of shape (m, ny) to shuffle.
            m is the same number of data points as in X.
            ny is the number of features.

    Returns:
        tuple: (X_shuffled, Y_shuffled)
            X_shuffled (numpy.ndarray): The shuffled X matrix.
            Y_shuffled (numpy.ndarray): The shuffled Y matrix.
    """
    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]
