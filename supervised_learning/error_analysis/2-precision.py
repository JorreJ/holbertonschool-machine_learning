#!/usr/bin/env python3
"""Module to calculate the precision for each class in a confusion matrix."""

import numpy as np


def precision(confusion):
    """Calculate the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the precision
            of each class.
    """
    tp = np.diag(confusion)
    pred = np.sum(confusion, axis=0)
    return tp / pred
