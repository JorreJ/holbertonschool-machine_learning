#!/usr/bin/env python3

"""
This module provides a function to add two arrays element-wise.

Both arrays must have the same length.
"""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.

    Each element of the resulting list is the sum of the elements
    at the same index in the input arrays.

    Args:
        arr1 (list): The first list of numbers.
        arr2 (list): The second list of numbers.

    Returns:
        list: A new list containing the element-wise sums.
        None: If the input lists do not have the same length.
    """
    if len(arr1) != len(arr2):
        return None

    new_list = []
    for x in range(len(arr1)):
        new_list.append(arr1[x] + arr2[x])

    return new_list
