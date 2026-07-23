#!/usr/bin/env python3
"""Module that perfoms K-means clustering."""

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


def kmeans(X, k, iterations=1000):
    """Perform K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): A 2D array of shape (n, d) containing the dataset.
        k (int): A positive integer containing the number of clusters.
        iterations (int): A positive integer containing the maximum number
            of iterations that should be performed. Defaults to 1000.

    Returns:
        tuple: A tuple (C, clss) where:
            - C (numpy.ndarray or None): A 2D array of shape (k, d) containing
              the centroid locations for each cluster, or None on failure.
            - clss (numpy.ndarray or None): A 1D array of shape (n,) containing
              the index of the cluster to which each data point belongs,
              or None on failure.
    """
    if (
            not isinstance(X, np.ndarray)
            or X.ndim != 2
            or X.shape[0] == 0
            or X.shape[1] == 0
            or not isinstance(k, int)
            or k < 1
            or not isinstance(iterations, int)
            or iterations < 1
    ):
        return None, None

    C = initialize(X, k)

    for i in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_new = np.array([X[clss == j].mean(axis=0)
                          if np.any(clss == j)
                          else initialize(X, 1)[0]
                          for j in range(k)])

        if np.array_equal(C, C_new):
            return C_new, clss

        C = C_new

    return C, clss
