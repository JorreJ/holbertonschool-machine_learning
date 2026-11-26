#!/usr/bin/env python3
"""Module that performs K-means clustering on a dataset using scikit-learn."""

import sklearn.cluster


def kmeans(X, k):
    """Perform K-means clustering on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) containing the data points
            to cluster.
        k (int): Number of clusters to form.

    Returns:
        tuple: A tuple containing:
            - C (numpy.ndarray): Array of shape (k, d) containing the centroid
              coordinates of each cluster.
            - clss (numpy.ndarray): Array of shape (n,) containing the index of
              the cluster each data point belongs to.
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    clss = model.fit_predict(X)
    C = model.cluster_centers_
    return C, clss
