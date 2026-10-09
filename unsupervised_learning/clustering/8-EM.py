#!/usr/bin/env python3
"""Expectation-Maximization algorithm for a Gaussian Mixture Model."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """Perform the expectation maximization algorithm for a GMM.

    Args:
        X (numpy.ndarray): 2D array of shape (n, d) containing the data set.
        k (int): Positive integer indicating the number of clusters.
        iterations (int, optional): Maximum number of iterations for the
            algorithm. Defaults to 1000.
        tol (float, optional): Non-negative float containing the tolerance on
            the log likelihood to stop early. Defaults to 1e-5.
        verbose (bool, optional): Determines if the algorithm prints
            information about the log likelihood every 10 iterations and after
            the last iteration. Defaults to False.

    Returns:
        tuple:
            - pi (numpy.ndarray): 1D array of shape (k,) containing the updated
              priors for each cluster.
            - m (numpy.ndarray): 2D array of shape (k, d) containing
              the updated centroid means for each cluster.
            - S (numpy.ndarray): 3D array of shape (k, d, d) containing the
              updated covariance matrices for each cluster.
            - g (numpy.ndarray): 2D array of shape (k, n) containing the
              posterior probabilities for each data point in each cluster.
            - l (float): The total log likelihood of the model.
            Returns (None, None, None, None, None) on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or X.shape[0] == 0 or X.shape[1] == 0
            or not isinstance(k, int) or k <= 0
            or not isinstance(iterations, int) or iterations <= 0
            or not isinstance(tol, (int, float)) or tol < 0
            or not isinstance(verbose, bool)):
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None
    g, l = expectation(X, pi, m, S)  # noqa: E741
    if g is None or l is None:
        return None, None, None, None, None
    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(l))
    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None
        g, new_l = expectation(X, pi, m, S)
        if g is None or new_l is None:
            return None, None, None, None, None
        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {:.5f}".format(
                i, new_l
            ))
        if abs(new_l - l) <= tol:
            if verbose and i % 10 != 0 and i != iterations:
                print("Log Likelihood after {} iterations: {:.5f}".format(
                    i, new_l
                ))
            l = new_l  # noqa: E741
            break
        l = new_l  # noqa: E741
    return pi, m, S, g, l
