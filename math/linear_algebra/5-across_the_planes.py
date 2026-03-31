#!/usr/bin/env python3

"""
This module provides a function to add two 2D matrices element-wise.

Both matrices must have the same dimensions.
"""

matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Each element of the resulting matrix is the sum of the elements
    at the same position in the input matrices.

    Args:
        mat1 (list of list): The first 2D matrix.
        mat2 (list of list): The second 2D matrix.

    Returns:
        list of list: A new matrix containing the element-wise sums.
        None: If the input matrices do not have the same shape.
    """
    if not mat1 or not mat1[0] or matrix_shape(mat1) != matrix_shape(mat2):
        return None

    new_mat = []
    temp = []

    for x in range(len(mat1)):
        for y in range(len(mat1[x])):
            temp.append(mat1[x][y] + mat2[x][y])
        new_mat.append(temp)
        temp = []

    return new_mat
