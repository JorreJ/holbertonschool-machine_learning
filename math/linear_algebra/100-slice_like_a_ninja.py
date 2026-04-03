#!/usr/bin/env python3

"""This module provides a function to slice an array along specified axes."""


def np_slice(matrix, axes={}):
    """
    Slice a NumPy array along given axes.

    The function allows selective slicing by specifying start, stop,
    and step values for each axis.

    Args:
        matrix (numpy.ndarray): The NumPy array to slice.
        axes (dict, optional): A dictionary where keys are axis indices
                               and values are tuples of the form
                               (start, stop, step) for slicing.

    Returns:
        numpy.ndarray: A sliced view of the original array.
    """
    new_mat = []
    slices = [slice(None)] * matrix.ndim

    for axis, instructions in axes.items():
        slices[axis] = slice(*instructions)

    new_mat = matrix[tuple(slices)]
    return new_mat
