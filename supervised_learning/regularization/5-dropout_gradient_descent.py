#!/usr/bin/env python3
"""Module to update weights and biases using gradient descent with Dropout."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update the weights of a neural network with Dropout regularization.

    Args:
        Y (numpy.ndarray): One-hot encoded labels of shape (classes, m).
            m is the number of data points.
        weights (dict): Dictionary of weights and biases of the network.
        cache (dict): Dictionary of outputs and dropout masks of each layer.
        alpha (float): The learning rate.
        keep_prob (float): The probability that a node will be kept.
        L (int): The number of layers of the network.
    """
    m = Y.shape[1]
    weights_copy = {key: np.copy(val) for key, val in weights.items()}

    A_L = cache["A" + str(L)]
    dZ = A_L - Y

    A_prev = cache["A" + str(L - 1)]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    weights["W" + str(L)] -= alpha * dW
    weights["b" + str(L)] -= alpha * db

    for x in range(L - 1, 0, -1):
        A_current = cache["A" + str(x)]
        D_current = cache["D" + str(x)]
        W_next = weights_copy["W" + str(x + 1)]

        dA = np.dot(W_next.T, dZ)

        dA = (dA * D_current) / keep_prob

        dZ = dA * (1 - A_current ** 2)

        A_prev = cache["A" + str(x - 1)]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights["W" + str(x)] -= alpha * dW
        weights["b" + str(x)] -= alpha * db
