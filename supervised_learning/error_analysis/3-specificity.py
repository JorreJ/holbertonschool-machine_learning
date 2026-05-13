#!/usr/bin/env python3
"""Module to calculate the specificity for each class in a confusion matrix."""

import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels.

    Returns:
        numpy.ndarray: An array of shape (classes,) containing the specificity
            of each class.
    """
    total = np.sum(confusion)
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = total - (tp + fn + fp)
    return tn / (tn + fp)
