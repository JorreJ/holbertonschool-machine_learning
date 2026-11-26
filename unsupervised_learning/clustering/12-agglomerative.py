#!/usr/bin/env python3
"""Module that performs agglomerative clustering and plots a dendrogram."""

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """Perform agglomerative clustering on a dataset and plot a dendrogram.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) containing the data points
            to cluster.
        dist (float): Maximum cophenetic distance for all clusters.

    Returns:
        numpy.ndarray: Array of shape (n,) containing the cluster indices for
        each data point.
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(
        Z,
        color_threshold=dist
    )
    plt.show()
    clss = scipy.cluster.hierarchy.fcluster(
        Z,
        t=dist,
        criterion='distance'
    )
    return clss
