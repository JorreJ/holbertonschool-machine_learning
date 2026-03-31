#!/usr/bin/env python3

"""
This module provides a function to transpose a 2D matrix.

A matrix transpose converts rows into columns.
"""


def matrix_transpose(matrix):
    """
    Transpose a 2D matrix.

    The element at position [i][j] in the original matrix
    becomes the element at position [j][i] in the transposed matrix.

    Args:
        matrix (list of list): A 2D list where each
        inner list represents a row.

    Returns:
        list of list: A new matrix representing the
        transpose of the input matrix.
    """
    temp = []
    new_matrix = []

    for x in range(len(matrix[0])):
        for line in matrix:
            temp.append(line[x])
        new_matrix.append(temp)
        temp = []

    return new_matrix
