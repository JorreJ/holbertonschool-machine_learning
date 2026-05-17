#!/usr/bin/env python3
"""Module to perform forward propagation with dropout regularization."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagate a neural network using dropout regularization.

    Args:
        X (numpy.ndarray): The input data of shape (nx, m).
            nx is the number of input features.
            m is the number of data points.
        weights (dict): A dictionary of the weights and biases of the network.
        L (int): The number of layers in the neural network.
        keep_prob (float): The probability that a node will be kept.

    Returns:
        dict: A dictionary containing the activation values of each layer
            and the dropout masks used in the hidden layers.
    """
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        Z = np.dot(W, A_prev) + b
        if i == L:
            t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache["A" + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache["A" + str(i)] = A
            cache["D" + str(i)] = D
    return cache
