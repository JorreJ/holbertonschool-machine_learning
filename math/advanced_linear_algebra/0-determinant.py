#!/usr/bin/env python3
"""Module that provides matrix operation functions."""

import numpy as np


def determinant(matrix):
    """Calculate the determinant of a square matrix.

    Args:
        matrix (list of lists): The matrix whose determinant is to be computed.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix.

    Returns:
        int: The determinant of the matrix.
    """
    if (len(matrix) == 0 or not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    return int(np.round(np.linalg.det(matrix)))
