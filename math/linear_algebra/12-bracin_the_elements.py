#!/usr/bin/env python3

"""This module performs element-wise operations on two NumPy arrays."""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise arithmetic operations on two NumPy arrays.

    The function returns the results of addition, subtraction,
    multiplication, and division applied element-wise.

    Args:
        mat1 (numpy.ndarray): The first NumPy array.
        mat2 (numpy.ndarray): The second NumPy array.

    Returns:
        tuple: A tuple containing four NumPy arrays:
               (addition, subtraction, multiplication, division).
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
