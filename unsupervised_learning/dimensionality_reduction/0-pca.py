#!/usr/bin/env python3
"""Principal Component Analysis."""
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset."""
    _, S, Vh = np.linalg.svd(X)
    cumulative_variance = np.cumsum(S) / np.sum(S)
    nd = np.argmax(cumulative_variance >= var) + 1
    return Vh[:nd].T
