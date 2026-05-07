#!/usr/bin/env python3
"""Module to update variables using RMSProp optimization algorithm."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The second moment weight.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of the variable.
        s (numpy.ndarray): The previous second moment of the variable.

    Returns:
        tuple: (new_var, new_s)
            new_var (numpy.ndarray): The updated variable.
            new_s (numpy.ndarray): The new second moment.
    """
    new_s = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    new_var = var - alpha * (grad / (np.sqrt(new_s) + epsilon))
    return new_var, new_s
