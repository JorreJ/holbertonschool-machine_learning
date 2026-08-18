#!/usr/bin/env python3
"""Module that defines a simple Recurrent Neural Network (RNN) cell."""

import numpy as np


class RNNCell:
    """Represent a simple Recurrent Neural Network (RNN) cell."""

    def __init__(self, i, h, o):
        """Initialize the RNNCell.

        Args:
            i (int): Dimension of the input data.
            h (int): Dimension of the hidden state.
            o (int): Dimension of the output data.
        """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

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
        h_next = np.tanh(x @ self.Wh + self.bh)
        Z = h_next @ self.Wy + self.by
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return h_next, y
