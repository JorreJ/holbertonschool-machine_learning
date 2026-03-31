#!/usr/bin/env python3

"""
This module provides a function to concatenate two 2D matrices.

Both matrices must be compatible in size for the chosen axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two 2D matrices along the specified axis.

    If axis=0, the matrices are stacked vertically (rows). The number
    of columns in both matrices must be the same.

    If axis=1, the matrices are stacked horizontally (columns). The number
    of rows in both matrices must be the same.

    Args:
        mat1 (list of list): The first 2D matrix.
        mat2 (list of list): The second 2D matrix.
        axis (int, optional): The axis along which to concatenate.
                              0 for rows, 1 for columns. Default is 0.

    Returns:
        list of list: A new matrix resulting from concatenation.
        None: If the matrices are incompatible for the chosen axis.
    """
    if ((axis == 0 and len(mat1[0]) != len(mat2[0]))
            or (axis == 1 and len(mat1) != len(mat2))):
        return

    if axis == 0:
        return mat1 + mat2

    if axis == 1:
        new_mat = []
        for x in range(len(mat1)):
            new_mat.append(mat1[x] + mat2[x])
        return new_mat
