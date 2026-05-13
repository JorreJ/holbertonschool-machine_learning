#!/usr/bin/env python3
"""Module to calculate the sensitivity for each class in a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """Calculate the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the sensitivity
            of each class.
    """
    tp = np.diag(confusion)
    real_positives = np.sum(confusion, axis=1)
    return tp / real_positives
