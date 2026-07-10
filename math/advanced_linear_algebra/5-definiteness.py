#!/usr/bin/env python3
"""Module that provides matrix operation and analysis functions."""

import numpy as np


def sub_matrix(matrix, row, column):
    """Extract a sub-matrix by removing a specific row and column.

    Args:
        matrix (list of lists): The original matrix.
        row (int): The index of the row to remove.
        column (int): The index of the column to remove.

    Returns:
        list of lists: The resulting sub-matrix.
    """
    return [
        [matrix[i][j] for j in range(len(matrix[i])) if j != column]
        for i in range(len(matrix)) if i != row
    ]


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

    det = 0
    for j in range(n):
        sign = 1 if j % 2 == 0 else -1
        coeff = matrix[0][j]
        sub_mat = sub_matrix(matrix, 0, j)
        det += sign * coeff * determinant(sub_mat)

    return det


def definiteness(matrix):
    """Determine the definiteness of a symmetric matrix.

    Args:
        matrix (numpy.ndarray): The matrix to analyze.

    Raises:
        TypeError: If matrix is not a numpy.ndarray.

    Returns:
        str or None: A string representing the definiteness type,
            or None if the matrix is not square or not symmetric.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        return None
    eigenvalues = np.linalg.eigvalsh(matrix)
    tol = 1e-12
    eigenvalues[np.abs(eigenvalues) < tol] = 0.0

    all_positive = np.all(eigenvalues > 0)
    all_negative = np.all(eigenvalues < 0)
    all_non_negative = np.all(eigenvalues >= 0)
    all_non_positive = np.all(eigenvalues <= 0)
    has_zero = np.any(eigenvalues == 0)

    if all_positive:
        return "Positive definite"
    elif all_negative:
        return "Negative definite"
    elif all_non_positive and has_zero:
        return "Negative semi-definite"
    elif all_non_negative and has_zero:
        return "Positive semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
