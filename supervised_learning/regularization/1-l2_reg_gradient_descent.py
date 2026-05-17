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
    m = Y.shape[1]
    weights_copy = {key: np.copy(val) for key, val in weights.items()}
    A_L = cache["A" + str(L)]
    dZ = A_L - Y
    A_prev = cache["A" + str(L - 1)]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    weights["b" + str(L)] -= alpha * db
    weights["W" + str(L)] -= alpha * (dW + (lambtha / m) *
                                      weights_copy["W" + str(L)])

    for x in range(L - 1, 0, -1):
        A_current = cache["A" + str(x)]
        W_next = weights_copy["W" + str(x + 1)]
        dZ = np.dot(W_next.T, dZ) * (1 - A_current ** 2)
        A_prev = cache["A" + str(x - 1)]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights["b" + str(x)] -= alpha * db
        weights["W" + str(x)] -= alpha * (dW + (lambtha / m) *
                                          weights_copy["W" + str(x)])
