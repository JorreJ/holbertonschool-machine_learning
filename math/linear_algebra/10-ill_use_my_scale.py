#!/usr/bin/env python3

"""This module provides a function to get the shape of a NumPy array."""


def np_shape(matrix):
    """
    Get the shape of a NumPy array.

    The shape describes the dimensions of the array (number of rows,
    columns, etc.).

    Args:
        matrix (numpy.ndarray): A NumPy array.

    Returns:
        tuple: A tuple representing the dimensions of the array.
    """
    return matrix.shape
