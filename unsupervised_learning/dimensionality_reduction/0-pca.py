#!/usr/bin/env python3
"""Module that performs Principal Component Analysis (PCA) on a dataset."""

import numpy as np


def pca(X, var=0.95):
    """Perform Principal Component Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number of
            data points and d is the number of dimensions. All dimensions are
            assumed to have a mean of 0.
        var (float): Fraction of the variance that the PCA transformation
            should maintain. Defaults to 0.95.

    Returns:
        numpy.ndarray: Weights matrix W of shape (d, ndim) that maintains var
        fraction of X's original variance.
    """
    covariance = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)
    ndim = np.searchsorted(cumulative_variance, var) + 1
    W = eigenvectors[:, :ndim]
    return W
