#!/usr/bin/env python3
"""Maximization step of the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """Calculate updated GMM parameters for the maximization step.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the data set.
        g (numpy.ndarray): 2D array of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.

    Returns:
        tuple:
            - pi (numpy.ndarray): 1D array of shape (k,) containing the updated
              priors for each cluster.
            - m (numpy.ndarray): 2D array of shape (k, d) containing
              the updated centroid means for each cluster.
            - S (numpy.ndarray): 3D array of shape (k, d, d) containing the
              updated covariance matrices for each cluster.
            Returns (None, None, None) on failure.
    """
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(g, np.ndarray)
        or g.ndim != 2
        or g.shape[0] == 0
        or g.shape[1] != X.shape[0]
        or not np.all(np.isfinite(X))
        or not np.all(np.isfinite(g))
        or np.any(g < 0)
    ):
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    N = np.sum(g, axis=1)
    if np.any(N <= 0):
        return None, None, None
    pi = N / n
    m = g @ X / N[:, np.newaxis]
    diff = X[np.newaxis, :, :] - m[:, np.newaxis, :]
    S = np.einsum('kn,kni,knj->kij', g, diff, diff)
    S = S / N[:, np.newaxis, np.newaxis]
    return pi, m, S
