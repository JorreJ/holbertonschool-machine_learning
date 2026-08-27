#!/usr/bin/env python3
"""Module that defines a Long Short-Term Memory (LSTM) cell."""

import numpy as np


def sigmoid(x):
    """Compute the sigmoid activation function for a given input.

    Args:
        x (numpy.ndarray or float): Input array or scalar.

    Returns:
        numpy.ndarray or float: Sigmoid of the input.
    """
    return 1.0 / (1.0 + np.exp(-x))


class LSTMCell:
    """Represent a Long Short-Term Memory (LSTM) cell."""

    def __init__(self, i, h, o):
        """Initialize the LSTMCell.

        Args:
            i (int): Dimension of the input data.
            h (int): Dimension of the hidden state.
            o (int): Dimension of the output data.
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step in an LSTM cell.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            c_prev (numpy.ndarray): Previous cell state of shape (m, h).
            x_t (numpy.ndarray): Input data for the current time step
                of shape (m, i).

        Returns:
            tuple:
                - h_next (numpy.ndarray): Next hidden state of shape (m, h).
                - c_next (numpy.ndarray): Next cell state of shape (m, h).
                - y (numpy.ndarray): Output prediction (softmax activation)
                  of shape (m, o).
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        f = sigmoid(x @ self.Wf + self.bf)
        u = sigmoid(x @ self.Wu + self.bu)
        c_bar = np.tanh(x @ self.Wc + self.bc)
        o = sigmoid(x @ self.Wo + self.bo)

        c_next = f * c_prev + u * c_bar
        h_next = o * np.tanh(c_next)

        Z = h_next @ self.Wy + self.by
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        return h_next, c_next, y
