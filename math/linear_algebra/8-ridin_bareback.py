#!/usr/bin/env python3

"""
This module provides a function to perform matrix multiplication.

Both matrices must be 2D matrices and have compatible dimensions.
"""


def mat_mul(mat1, mat2):
    """
    Multiply two 2D matrices.

    Matrix multiplication is performed by taking the dot product of rows
    from the first matrix and columns from the second matrix.

    The number of columns in the first matrix must be equal to the number
    of rows in the second matrix.

    Args:
        mat1 (list of list): The first 2D matrix.
        mat2 (list of list): The second 2D matrix.

    Returns:
        list of list: A new matrix resulting from the multiplication.
        None: If the matrices cannot be multiplied
            due to incompatible dimensions.
    """
    if len(mat1[0]) != len(mat2):
        return None

    new_mat = []
    temp = []
    number = 0

    for x in range(len(mat1)):
        for y in range(len(mat2[0])):
            for z in range(len(mat1[0])):
                number += (mat1[x][z] * mat2[z][y])
            temp.append(number)
            number = 0
        new_mat.append(temp)
        temp = []

    return new_mat
