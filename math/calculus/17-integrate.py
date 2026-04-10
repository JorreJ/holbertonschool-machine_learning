#!/usr/bin/env python3

"""Computes the integral of a polynomial as a list of coefficients."""


def poly_integral(poly, C=0):
    """
    Compute the integral of a polynomial.

    The polynomial is represented as a list where the index represents
    the power of x and the value at that index is the coefficient.
    For example: [a, b, c] represents a + bx + cx².

    The integral increases the power of each term by 1 and divides
    the coefficient accordingly. A constant of integration C is added
    as the first element of the resulting list.

    Args:
        poly (list): A list of coefficients of the polynomial.
        C (int, optional): The constant of integration. Default is 0.

    Returns:
        list: A new list of coefficients representing the integral.
        None: If the input is invalid.
    """
    if not poly or not isinstance(poly, list) or not isinstance(C, int):
        return None

    new_list = [C]

    for i in range(len(poly)):
        temp = poly[i] / (i + 1)
        if temp.is_integer():
            temp = int(temp)
        new_list.append(temp)

    while len(new_list) > 1 and new_list[-1] == 0:
        new_list.pop()

    return new_list
