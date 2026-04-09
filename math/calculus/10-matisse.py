#!/usr/bin/env python3

"""Calculates the derivative of a polynomial by a list of coefficients."""


def poly_derivative(poly):
    """
    Compute the derivative of a polynomial.

    The polynomial is represented as a list where the index represents
    the power of x and the value at that index is the coefficient.
    For example: [a, b, c] represents a + bx + cx².

    Args:
        poly (list): A list of coefficients of the polynomial.

    Returns:
        list: A new list of coefficients representing the derivative.
        None: If the input is not a valid list or is empty.
    """
    if not poly or not isinstance(poly, list):
        return None

    new_list = []
    for x in range(1, len(poly)):
        new_list.append(poly[x] * x)

    return new_list
