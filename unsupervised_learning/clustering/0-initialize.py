#!/usr/bin/env python3
"""Module that provides initialization for K-means clustering."""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means.

    Args:
        X (numpy.ndarray): A 2D array of shape (n, d) containing the dataset
            that will be used for K-means clustering.
        k (int): A positive integer containing the number of clusters.

    Returns:
        numpy.ndarray or None: A 2D array of shape (k, d) containing the
            initialized centroids for each cluster, or None on failure.
    """
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(k, int)
        or k < 1
    ):
        return None

    n, d = X.shape

    centroids = np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, d)
    )

    return centroids
