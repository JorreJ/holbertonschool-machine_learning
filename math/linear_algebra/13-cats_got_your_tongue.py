#!/usr/bin/env python3
"""This module concatenates two NumPy arrays along a specified axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two NumPy arrays along a given axis.

    If axis=0, arrays are concatenated along rows.
    If axis=1, arrays are concatenated along columns.

    Args:
        mat1 (numpy.ndarray): The first NumPy array.
        mat2 (numpy.ndarray): The second NumPy array.
        axis (int, optional): The axis along which the arrays will be joined.
                              Default is 0.

    Returns:
        numpy.ndarray: A new array resulting from the concatenation.

    Raises:
        ValueError: If the arrays have incompatible shapes for concatenation.
    """
    new_mat = np.concatenate((mat1, mat2), axis)
    return new_mat
