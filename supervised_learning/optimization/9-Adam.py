#!/usr/bin/env python3
"""Module to update variables using Adam optimization algorithm."""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable using the Adam optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight (first moment).
        beta2 (float): The RMSProp weight (second moment).
        epsilon (float): A small constant to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of the variable.
        v (numpy.ndarray): The previous first moment of the variable.
        s (numpy.ndarray): The previous second moment of the variable.
        t (int): The time step used for bias correction.

    Returns:
        tuple: (new_var, new_v, new_s)
            new_var (numpy.ndarray): The updated variable.
            new_v (numpy.ndarray): The new first moment.
            new_s (numpy.ndarray): The new second moment.
    """
    new_v = (beta1 * v) + ((1 - beta1) * grad)
    new_s = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    corrected_v = new_v / (1 - (beta1 ** t))
    corrected_s = new_s / (1 - (beta2 ** t))
    new_var = var - alpha * (corrected_v / (np.sqrt(corrected_s) + epsilon))
    return new_var, new_v, new_s
