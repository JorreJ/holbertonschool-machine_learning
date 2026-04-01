#!/usr/bin/env python3

"""This module provides a function to transpose a NumPy array."""


def np_transpose(matrix):
    """
    Transpose a NumPy array.

    The transpose of an array is obtained by swapping its axes.
    For a 2D array, this converts rows into columns.

    Args:
        matrix (numpy.ndarray): A NumPy array.

    Returns:
        numpy.ndarray: The transposed array.
    """
    return matrix.transpose()
