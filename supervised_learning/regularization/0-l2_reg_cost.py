#!/usr/bin/env python3
"""Module to calculate the cost of a neural network with L2 regularization."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculate the cost of a neural network with L2 regularization.

    Args:
        cost (float): The cost of the network without L2 regularization.
        lambtha (float): The regularization parameter.
        weights (dict): A dictionary of the weights of the neural network.
        L (int): The number of layers in the neural network.
        m (int): The number of data points used.

    Returns:
        float: The cost of the network accounting for L2 regularization.
    """
    l2_sum = 0
    for i in range(1, L + 1):
        key = 'W' + str(i)
        l2_sum += np.sum(np.square(weights[key]))
    new_cost = l2_sum * (lambtha / (2 * m)) + cost
    return new_cost
