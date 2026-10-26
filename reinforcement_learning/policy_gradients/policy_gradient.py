#!/usr/bin/env python3
"""Module that computes policy probabilities using softmax."""

import numpy as np


def policy(matrix, weight):
    """Compute the softmax policy probabilities for a set of states.

    Args:
        matrix (numpy.ndarray): 2D array of shape (n, d) containing the state
            features.
        weight (numpy.ndarray): 2D array of shape (d, a) containing the policy
            weights.

    Returns:
        numpy.ndarray: 2D array of shape (n, a) containing the policy
        probabilities for each state and action.
    """
    z = matrix @ weight
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)
