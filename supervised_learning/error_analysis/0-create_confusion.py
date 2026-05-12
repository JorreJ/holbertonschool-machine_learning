#!/usr/bin/env python3
"""Module to create a confusion matrix using numpy."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix.

    Args:
        labels (numpy.ndarray): One-hot encoded labels of shape (m, n).
            m is the number of data points.
            n is the number of classes.
        logits (numpy.ndarray): Predicted one-hot labels of shape (m, n).

    Returns:
        numpy.ndarray: A confusion matrix of shape (n, n) with row indices
            representing the correct labels and column indices representing
            the predicted labels.
    """
    n = labels.shape[1]
    real_index = np.argmax(labels, axis=1)
    predicted_index = np.argmax(logits, axis=1)
    combined_index = real_index * n + predicted_index
    conf_matrix = np.bincount(combined_index, minlength=n**2).reshape(n, n)
    return conf_matrix.astype(float)
