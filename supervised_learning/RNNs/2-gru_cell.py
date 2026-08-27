#!/usr/bin/env python3
"""Module that defines a Gated Recurrent Unit (GRU) cell."""

import numpy as np


def sigmoid(x):
    """Compute the sigmoid activation function for a given input.

    Args:
        x (numpy.ndarray or float): Input array or scalar.

    Returns:
        numpy.ndarray or float: Sigmoid of the input.
    """
    return 1 / (1 + np.exp(-x))


class GRUCell:
    """Represent a Gated Recurrent Unit (GRU) cell."""

    def __init__(self, i, h, o):
        """Initialize the GRUCell.

        Args:
            i (int): Dimension of the input data.
            h (int): Dimension of the hidden state.
            o (int): Dimension of the output data.
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step in a GRU cell.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            x_t (numpy.ndarray): Input data for the current time step
                of shape (m, i).

        Returns:
            tuple:
                - h_next (numpy.ndarray): Next hidden state of shape (m, h).
                - y (numpy.ndarray): Output prediction (softmax activation)
                  of shape (m, o).
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(x @ self.Wz + self.bz)
        r = sigmoid(x @ self.Wr + self.br)
        reset_h = r * h_prev
        h = np.concatenate((reset_h, x_t), axis=1)
        h_tilde = np.tanh((h @ self.Wh + self.bh))
        h_next = (1 - z) * h_prev + z * h_tilde
        Z = h_next @ self.Wy + self.by
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return h_next, y
