#!/usr/bin/env python3
"""Module that performs forward propagation for a deep RNN."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN over multiple time steps.

    Args:
        rnn_cells (list): List of RNNCell (or compatible) instances used for
            each layer.
        X (numpy.ndarray): Input data of shape (t, m, i) where:
            - t is the number of time steps.
            - m is the batch size.
            - i is the dimensionality of the inputs.
        h_0 (numpy.ndarray): Initial hidden state of shape (l, m, h) where:
            - l is the number of layers.
            - m is the batch size.
            - h is the dimensionality of the hidden state.

    Returns:
        tuple:
            - H (numpy.ndarray): All hidden states of shape (t + 1, l, m, h).
            - Y (numpy.ndarray): All outputs of shape (t, m, o).
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0
    h_k = h_0

    for k in range(t):
        h_next = np.zeros_like(h_k)
        x = X[k]
        for j in range(l):
            h, y = rnn_cells[j].forward(h_k[j], x)

            H[k + 1, j] = h

            h_next[j] = h
            x = h
        Y[k] = y
        h_k = h_next

    return H, Y
