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
