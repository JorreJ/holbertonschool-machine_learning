#!/usr/bin/env python3

"""This module computes the summation of squares of integers from 1 to n."""


def summation_i_squared(n):
    """
    Recursively compute the sum of the squares of integers from 1 to n.

    The function calculates:
        1^2 + 2^2 + 3^2 + ... + n^2

    Args:
        n (int): The upper bound of the summation.

    Returns:
        int: The sum of squares from 1 to n.
        None: If n is less than 1.
    """
    if n > 1:
        return (n**2) + summation_i_squared(n - 1)
    elif n == 1:
        return 1
    else:
        return None
