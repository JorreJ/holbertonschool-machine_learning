#!/usr/bin/env python3

"""This module performs matrix multiplication between two NumPy arrays."""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Multiply two NumPy arrays using matrix multiplication.

    The number of columns in the first array must equal the number of rows
    in the second array. This performs the standard linear algebraic
    matrix multiplication.

    Args:
        mat1 (numpy.ndarray): The first NumPy array.
        mat2 (numpy.ndarray): The second NumPy array.

    Returns:
        numpy.ndarray: A new array resulting from the matrix multiplication.

    Raises:
        ValueError: If the arrays have incompatible shapes for multiplication.
    """
    new_mat = np.matmul(mat1, mat2)
    return new_mat
