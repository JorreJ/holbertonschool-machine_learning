#!/usr/bin/env python3
"""Module for fitting a Gaussian Mixture Model using scikit-learn."""
import sklearn.mixture


def gmm(X, k):
    """
    Calculate a Gaussian Mixture Model (GMM) from a dataset.

    Args:
        X (numpy.ndarray): Shape (n, d) containing the dataset.
        k (int): The number of clusters.

    Returns:
        tuple: (pi, m, S, clss, bic)
            - pi (numpy.ndarray): Shape (k,) containing the cluster priors.
            - m (numpy.ndarray): Shape (k, d) containing the centroid means.
            - S (numpy.ndarray): Covariance matrices of shape (k, d, d).
            - clss (numpy.ndarray): Shape (n,) containing the cluster indices.
            - bic (float): The Bayesian Information Criterion (BIC) value.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k)
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
