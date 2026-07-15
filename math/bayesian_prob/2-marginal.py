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


def intersection(x, n, P, Pr):
    """Calculate the intersection of having x successes with prior beliefs.

    Args:
        x (int): The number of successes.
        n (int): The total number of trials.
        P (numpy.ndarray): A 1D array containing various probabilities
            of success.
        Pr (numpy.ndarray): A 1D array containing prior beliefs.

    Raises:
        ValueError: If n is not a positive integer, if x is not an integer
            greater than or equal to 0, if x is greater than n, if any value
            in P or Pr is not in the range [0, 1], or if Pr does not sum to 1.
        TypeError: If P or Pr is not a 1D numpy.ndarray, or if Pr does not have
            the same shape as P.

    Returns:
        numpy.ndarray: A 1D array containing the intersection of likelihood
            and prior beliefs for each probability in P.
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any((P < 0) | (P > 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """Calculate the marginal probability of obtaining x successes in n trials.

    Args:
        x (int): The number of successes.
        n (int): The total number of trials.
        P (numpy.ndarray): A 1D array containing various probabilities
            of success.
        Pr (numpy.ndarray): A 1D array containing prior beliefs.

    Raises:
        ValueError: If n is not a positive integer, if x is not an integer
            greater than or equal to 0, if x is greater than n, if any value
            in P or Pr is not in the range [0, 1], or if Pr does not sum to 1.
        TypeError: If P or Pr is not a 1D numpy.ndarray, or if Pr does not have
            the same shape as P.

    Returns:
        float: The marginal probability of obtaining x successes.
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if np.any((P < 0) | (P > 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    return np.sum(intersection(x, n, P, Pr))
