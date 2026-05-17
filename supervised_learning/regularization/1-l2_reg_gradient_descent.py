#!/usr/bin/env python3
"""Module to update weights and biases using gradient descent with L2 reg."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update neural network weights and biases using gradient descent with L2.

    Args:
        Y (numpy.ndarray): One-hot encoded labels of shape (classes, m).
            m is the number of data points.
        weights (dict): Dictionary of weights and biases of the neural network.
        cache (dict): Dictionary of outputs of each layer of the network.
        alpha (float): The learning rate.
        lambtha (float): The L2 regularization parameter.
        L (int): The number of layers in the neural network.
    """
    m = len(Y[0])
    last_layer = cache["A" + str(L)]
    dZ_L = last_layer - Y
    previous_layer = cache["A" + str(L - 1)]
    dW_L = (1 / m) * np.dot(dZ_L, previous_layer.T)
    db_L = (1 / m) * np.sum(dZ_L, axis=1, keepdims=True)
    next_dZ = dZ_L
    weights["b" + str(L)] -= alpha * db_L
    weights["W" + str(L)] -= alpha * (dW_L +
                                      (lambtha / m * weights["W" + str(L)]))
    for x in range(L - 1, 0, -1):
        next_W = weights["W" + str(x + 1)]
        current_A = cache["A" + str(x)]
        dZ_x = np.dot(next_W.T, next_dZ) * (1 - np.square(current_A))
        previous_layer = cache["A" + str(x - 1)]
        dW_x = (1 / m) * np.dot(dZ_x, previous_layer.T)
        db_x = (1 / m) * np.sum(dZ_x, axis=1, keepdims=True)
        next_dZ = dZ_x
        weights["b" + str(x)] -= alpha * db_x
        weights["W" + str(x)] -= alpha * (dW_x + (lambtha / m
                                                  * weights["W" + str(x)]))
