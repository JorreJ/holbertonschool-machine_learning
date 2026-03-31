#!/usr/bin/env python3

"""
This module provides a function to concatenate two arrays.

The elements of the second array are appended to the first array.
"""


def cat_arrays(arr1, arr2):
    """
    Concatenate two arrays.

    Args:
        arr1 (list): The first list of elements.
        arr2 (list): The second list of elements.

    Returns:
        list: A new list containing all elements of
        arr1 followed by all elements of arr2.
    """
    return arr1 + arr2
