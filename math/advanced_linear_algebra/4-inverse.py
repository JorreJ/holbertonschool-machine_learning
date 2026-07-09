#!/usr/bin/env python3
"""Provides matrix operation functions including cofactor computation."""


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


def minor(matrix):
    """Calculate the minor matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix whose minor is to be computed.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The minor matrix.
    """
    if (len(matrix) == 0 or not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    n = len(matrix)
    if n == 1:
        return [[1]]
    minor_mat = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            sub_mat = sub_matrix(matrix, i, j)
            minor_mat[i][j] = determinant(sub_mat)

    return minor_mat


def cofactor(matrix):
    """Calculate the cofactor matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix whose cofactor matrix is to be
            computed.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The cofactor matrix.
    """
    if (len(matrix) == 0 or not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    minor_mat = minor(matrix)
    n = len(minor_mat)
    sign_mat = [[(-1) ** (i + j) for j in range(n)] for i in range(n)]
    cof_mat = [[minor_mat[i][j] * sign_mat[i][j]
                for j in range(n)] for i in range(n)]
    return cof_mat


def adjugate(matrix):
    """Calculate the adjugate matrix of a square matrix.

    Args:
        matrix (list of lists): The matrix whose adjugate matrix is to be
            computed.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists: The adjugate matrix.
    """
    if (len(matrix) == 0 or not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    cof_mat = cofactor(matrix)
    adj_mat = [[row[i] for row in cof_mat] for i in range(len(cof_mat[0]))]
    return adj_mat


def inverse(matrix):
    """Calculate the inverse of a square matrix.

    Args:
        matrix (list of lists): The matrix whose inverse is to be computed.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a non-empty square matrix.

    Returns:
        list of lists or None: The inverse matrix, or None if the matrix
            is singular (determinant is 0).
    """
    if (len(matrix) == 0 or not isinstance(matrix, list)
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    det = determinant(matrix)
    if det == 0:
        return None
    adj_mat = adjugate(matrix)
    inv_mat = [[x / det for x in row] for row in adj_mat]
    return inv_mat
