#!/usr/bin/env python3
"""Module that determines the optimum number of clusters for K-means."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Find cluster sizes by comparing their intra-cluster variance.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the dataset.
        kmin (int, optional): Minimum number of clusters to check.
            Defaults to 1.
        kmax (int, optional): Maximum number of clusters to check.
            Defaults to None.
        iterations (int, optional): Maximum number of iterations for K-means.
            Defaults to 1000.

    Returns:
        tuple:
            - results (list): List containing the outputs of K-means for each
              k in the range [kmin, kmax].
            - d_vars (list): List containing the difference in variance from
              the smallest cluster size for each k in the range [kmin, kmax].
            Returns (None, None) on failure.
    """
    if (not isinstance(X, np.ndarray)
            or X.ndim != 2
            or X.shape[0] == 0
            or X.shape[1] == 0
            or not isinstance(kmin, int)
            or kmin < 1
            or (kmax is not None
                and (not isinstance(kmax, int)
                     or kmax < 1
                     or kmax <= kmin))
            or not isinstance(iterations, int)
            or iterations < 1):
        return None, None
    n = X.shape[0]
    if kmax is None:
        kmax = n
    results = []
    d_vars = []
    first_var = None
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        var = variance(X, C)
        if var is None:
            return None, None
        if first_var is None:
            first_var = var
        d_vars.append(first_var - var)
    return results, d_vars
