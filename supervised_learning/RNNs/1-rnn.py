#!/usr/bin/env python3
"""Module that performs forward propagation for a simple RNN."""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN over multiple time steps.

    Args:
        rnn_cell: Instance of RNNCell (or compatible) used for forward steps.
        X (numpy.ndarray): Input data of shape (t, m, i) where:
            - t is the number of time steps.
            - m is the batch size.
            - i is the dimensionality of the inputs.
        h_0 (numpy.ndarray): Initial hidden state of shape (m, h) where:
            - m is the batch size.
            - h is the dimensionality of the hidden state.

    Returns:
        tuple:
            - H (numpy.ndarray): All hidden states of shape (t + 1, m, h).
            - Y (numpy.ndarray): All outputs of shape (t, m, o).
    """
    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0
    h_k = h_0

    for k in range(t):
        h_next, y = rnn_cell.forward(h_k, X[k])

        H[k + 1] = h_next
        Y[k] = y

        h_k = h_next

    return H, Y
