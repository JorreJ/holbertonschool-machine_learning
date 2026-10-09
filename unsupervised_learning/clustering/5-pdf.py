#!/usr/bin/env python3
"""Calculates the PDF of a Gaussian distribution."""

import numpy as np


def pdf(X, m, S):
    """Calculate the probability density function of a Gaussian distribution.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the data points
            whose PDF should be calculated.
        m (numpy.ndarray): 1D array of shape (d,) containing the mean of the
            distribution.
        S (numpy.ndarray): 2D array of shape (d, d) containing the covariance
            matrix of the distribution.

    Returns:
        numpy.ndarray: 1D array of shape (n,) containing the PDF values for
        each data point in X, or None on failure.
    """
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(m, np.ndarray)
        or m.shape != (X.shape[1],)
        or not isinstance(S, np.ndarray)
        or S.shape != (X.shape[1], X.shape[1])
    ):
        return None
    try:
        d = X.shape[1]
        det = np.linalg.det(S)
        if det <= 0:
            return None
        inv = np.linalg.inv(S)
        diff = X - m
        exponent = -0.5 * np.sum(
            (diff @ inv) * diff,
            axis=1
        )
        denom = np.sqrt((2 * np.pi) ** d * det)
        P = np.exp(exponent) / denom
        return np.maximum(P, 1e-300)
    except np.linalg.LinAlgError:
        return None
