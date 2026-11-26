#!/usr/bin/env python3
"""Module that reduces the dimensionality of a dataset using PCA."""

import numpy as np


def pca(X, ndim):
    """Perform Principal Component Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number of
            data points and d is the number of dimensions.
        ndim (int): New dimensionality of the transformed dataset.

    Returns:
        numpy.ndarray: Transformed dataset of shape (n, ndim).
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    _, _, Vt = np.linalg.svd(X_centered)
    W = Vt[:ndim].T
    return X_centered @ W
