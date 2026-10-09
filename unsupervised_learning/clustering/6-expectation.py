#!/usr/bin/env python3
"""Expectation step of the EM algorithm for a GMM."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate the expectation step of the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the data set.
        pi (numpy.ndarray): 1D array of shape (k,) containing the priors for
            each cluster.
        m (numpy.ndarray): 2D array of shape (k, d) containing the centroid
            means for each cluster.
        S (numpy.ndarray): 3D array of shape (k, d, d) containing the
            covariance matrices for each cluster.

    Returns:
        tuple:
            - g (numpy.ndarray): 2D array of shape (k, n) containing the
              posterior probabilities for each data point in each cluster.
            - l (float): The total log likelihood of the model.
            Returns (None, None) on failure.
    """
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(pi, np.ndarray)
        or pi.ndim != 1
        or not isinstance(m, np.ndarray)
        or m.ndim != 2
        or not isinstance(S, np.ndarray)
        or S.ndim != 3
    ):
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if (
        k == 0
        or pi.shape != (k,)
        or m.shape != (k, d)
        or S.shape != (k, d, d)
        or np.any(pi < 0)
        or not np.all(np.isfinite(X))
        or not np.all(np.isfinite(pi))
        or not np.all(np.isfinite(m))
        or not np.all(np.isfinite(S))
        or not np.isclose(np.sum(pi), 1)
    ):
        return None, None
    g = np.zeros((k, n))
    for j in range(k):
        P = pdf(X, m[j], S[j])
        if P is None or not np.all(np.isfinite(P)):
            return None, None
        g[j] = pi[j] * P
    total = np.sum(g, axis=0)
    if np.any(total <= 0) or not np.all(np.isfinite(total)):
        return None, None
    l = np.sum(np.log(total))
    g = g / total
    return g, l
