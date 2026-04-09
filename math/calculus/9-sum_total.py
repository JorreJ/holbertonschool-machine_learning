#!/usr/bin/env python3

"""Computes the summation of squares of integers from 1 to n."""


def summation_i_squared(n):
    """
    Compute the sum of the squares of integers from 1 to n.

    This function uses the formula:
        n(n + 1)(2n + 1) / 6

    Args:
        n (int): The upper bound of the summation. Must be a positive integer.

    Returns:
        int: The sum of squares from 1 to n.
        None: If n is not a positive integer.
    """
    if not isinstance(n, int) or not n > 0:
        return None

    return (n * (n + 1) * (2 * n + 1)) // 6
