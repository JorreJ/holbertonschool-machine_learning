#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification."""

import numpy as np


class Neuron:
    """Represents a single neuron performing binary classification.

    Attributes:
        W (numpy.ndarray): The weights vector for the neuron.
        b (float): The bias for the neuron.
        A (float): The activated output of the neuron (prediction).
    """

    def __init__(self, nx):
        """Initialize a new Neuron instance.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the weights vector for the neuron.

        Returns:
            numpy.ndarray: The weights vector.
        """
        return self.__W

    @property
    def b(self):
        """Get the bias for the neuron.

        Returns:
            float: The bias value.
        """
        return self.__b

    @property
    def A(self):
        """Get the activated output of the neuron.

        Returns:
            float: The activated output.
        """
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron.

        Updates the private attribute __A using a sigmoid activation function.

        Args:
            X (numpy.ndarray): The input data with shape (nx, m).
                nx is the number of input features to the neuron.
                m is the number of examples.

        Returns:
            float: The activated output of the neuron (__A).
        """
        temp = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-temp))
        return self.__A

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression.

        The cost function is the binary cross-entropy (logistic loss).

        Args:
            Y (numpy.ndarray): Correct labels for the input data, shape (1, m).
            A (numpy.ndarray): Activated output of the neuron for each example,
                shape (1, m).

        Returns:
            float: The computed cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions and cost.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
                nx is the number of input features to the neuron.
                m is the number of examples.
            Y (numpy.ndarray): Correct labels for the input data, shape (1, m).

        Returns:
            tuple: (prediction, cost)
                prediction (numpy.ndarray): Predicted labels (0 or 1) with
                    shape (1, m).
                cost (float): The computed cost for the predictions.
        """
        self.forward_prop(X)
        preds = (self.__A >= 0.5).astype(int)
        cost = self.cost(Y, self.__A)
        return (preds, cost)
