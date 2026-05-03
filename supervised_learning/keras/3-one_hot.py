#!/usr/bin/env python3
"""Module to convert label vectors into one-hot matrices."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Convert a numeric label vector into a one-hot matrix.

    Args:
        labels (numpy.ndarray): The numeric labels to be converted.
        classes (int): The total number of classes. If None, it is
            inferred from the labels.

    Returns:
        numpy.ndarray: The one-hot matrix.
    """
    return K.utils.to_categorical(labels, classes)
