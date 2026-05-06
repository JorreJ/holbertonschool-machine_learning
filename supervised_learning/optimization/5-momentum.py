#!/usr/bin/env python3
"""Module to update variables using Gradient Descent with Momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using the gradient descent with momentum algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of the variable.
        v (numpy.ndarray): The previous first moment of the variable.

    Returns:
        tuple: (new_var, new_v)
            new_var (numpy.ndarray): The updated variable.
            new_v (numpy.ndarray): The new first moment.
    """
    new_v = (beta1 * v) + ((1 - beta1) * grad)
    new_var = var - (alpha * new_v)
    return new_var, new_v
