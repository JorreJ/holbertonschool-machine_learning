#!/usr/bin/env python3

"""
This module provides a utility function
to determine the shape of a nested list (matrix).

It can be used to get the dimensions of any rectangular nested list structure.
"""


def matrix_shape(matrix):
    """Return the shape of a nested list (matrix) as a list of dimensions.

    It works by iteratively checking if the current object is a list.
    At each level, it appends the length of the current list to the
    'shape' list and then moves to the next nested level by selecting
    the first element, until it reaches a non-list element.
    """
    shape = []  # Initialize an empty list to store the dimensions
    while isinstance(matrix, list):
        shape.append(len(matrix))  # Append the length of the current level
        matrix = matrix[0]        # Move to the next nested level
    return shape  # Return the list of dimensions
