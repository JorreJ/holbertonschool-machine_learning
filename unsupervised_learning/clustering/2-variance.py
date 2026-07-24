#!/usr/bin/env python3
"""Module that calculates the total intra-cluster variance."""

import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance for a data set.

    Args:
        X (numpy.ndarray): A 2D array of shape (n, d) containing the dataset.
        C (numpy.ndarray): A 2D array of shape (k, d) containing the
            centroid locations for each cluster.

    Returns:
        float or None: The total variance (sum of squared distances to the
            nearest centroid), or None on failure.
    """
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(C, np.ndarray)
        or C.ndim != 2
        or C.shape[0] == 0
        or C.shape[1] == 0
        or X.shape[1] != C.shape[1]
    ):
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    dist_to_centroid = np.linalg.norm(X - C[clss], axis=1)
    return np.sum(dist_to_centroid ** 2)
