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
    covariance = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, indices]
    components = eigenvectors[:, :ndim]
    X_reduced = X_centered @ components
    return X_reduced
