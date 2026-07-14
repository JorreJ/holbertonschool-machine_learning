#!/usr/bin/env python3
"""Module that provides a function to calculate binomial likelihoods."""

import numpy as np


def likelihood(x, n, P):
    """Calculate the likelihood of obtaining x successes in n trials.

    Args:
        x (int): The number of successes.
        n (int): The total number of trials.
        P (numpy.ndarray): A 1D array containing various probabilities
            of success.

    Raises:
        ValueError: If n is not a positive integer, if x is not an integer
            greater than or equal to 0, if x is greater than n, or if any value
            in P is not in the range [0, 1].
        TypeError: If P is not a 1D numpy.ndarray.

    Returns:
        numpy.ndarray: A 1D array containing the likelihood for each
            probability in P.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError('x must be an integer that is greater than'
                         ' or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any((P < 0) | (P > 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    coeff = np.math.factorial(n) / (np.math.factorial(x)
                                    * np.math.factorial(n - x))
    return coeff * (P ** x) * ((1 - P) ** (n - x))
