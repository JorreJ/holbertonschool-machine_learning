#!/usr/bin/env python3
"""Module to find the optimal number of clusters for a GMM using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using BIC.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the data set.
        kmin (int, optional): Minimum number of clusters to check.
            Defaults to 1.
        kmax (int, optional): Maximum number of clusters to check.
            Defaults to None.
        iterations (int, optional): Maximum number of iterations for the
            EM algorithm. Defaults to 1000.
        tol (float, optional): Non-negative float containing the tolerance on
            the log likelihood to stop early. Defaults to 1e-5.
        verbose (bool, optional): Determines if the EM algorithm prints
            information about the log likelihood. Defaults to False.

    Returns:
        tuple:
            - best_k (int): Best value for k based on the lowest BIC.
            - best_result (tuple): Tuple of (pi, m, S) corresponding to the
              best model params.
            - l (numpy.ndarray): 1D array containing the log likelihood for
              each cluster size.
            - b (numpy.ndarray): 1D array containing the BIC values for each
              cluster size.
            Returns (None, None, None, None) on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or X.shape[0] == 0 or X.shape[1] == 0
            or not isinstance(kmin, int) or kmin <= 0
            or not isinstance(iterations, int) or iterations <= 0
            or not isinstance(tol, (int, float)) or tol < 0
            or not isinstance(verbose, bool)):
        return None, None, None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if (not isinstance(kmax, int) or kmax <= 0
            or kmin > kmax or kmax > n):
        return None, None, None, None
    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    best_k = None
    best_result = None
    best_bic = np.inf
    for k in range(kmin, kmax + 1):
        result = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        pi, m, S, g, log_likelihood = result
        if (pi is None or m is None or S is None
                or g is None or log_likelihood is None):
            return None, None, None, None
        p = (k - 1) + k * d + k * d * (d + 1) // 2
        idx = k - kmin
        l[idx] = log_likelihood
        b[idx] = p * np.log(n) - 2 * log_likelihood
        if b[idx] < best_bic:
            best_bic = b[idx]
            best_k = k
            best_result = (pi, m, S)
    return best_k, best_result, l, b
