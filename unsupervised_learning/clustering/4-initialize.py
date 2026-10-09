#!/usr/bin/env python3
"""Module that initializes variables for a Gaussian Mixture Model."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize variables for a Gaussian Mixture Model (GMM).

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the dataset.
        k (int): Positive integer indicating the number of clusters.

    Returns:
        tuple:
            - pi (numpy.ndarray): 1D array of shape (k,) containing the
              priors for each cluster, initialized to 1/k.
            - m (numpy.ndarray): 2D array of shape (k, d) containing the
              centroid means initialized with K-means.
            - S (numpy.ndarray): 3D array of shape (k, d, d) containing the
              covariance matrices initialized as identity matrices.
            Returns (None, None, None) on failure.
    """
    if (not isinstance(X, np.ndarray)
            or X.ndim != 2
            or X.shape[0] == 0
            or X.shape[1] == 0
            or not isinstance(k, int)
            or k < 1):
        return None, None, None
    n, d = X.shape
    if k > n:
        return None, None, None
    pi = np.full(k, 1 / k)
    m, clss = kmeans(X, k)
    if m is None or clss is None:
        return None, None, None
    S = np.tile(np.eye(d), (k, 1, 1))
    return pi, m, S
