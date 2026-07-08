#!/usr/bin/env python3
"""Module that provides matrix operation functions."""


def determinant(matrix):
    """Calculate the determinant of a square matrix.

    Args:
        matrix (list of lists): The matrix whose determinant is to be computed.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square matrix.

    Returns:
        int or float: The determinant of the matrix.
    """
    if (len(matrix) == 0 or not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    if n == 3:
        (a, b, c), (d, e, f), (g, h, i) = matrix
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
