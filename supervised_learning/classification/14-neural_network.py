#!/usr/bin/env python3
"""Module that defines a neural network with one hidden layer."""

import numpy as np


class NeuralNetwork:
    """Represents a neural network with one hidden layer.

    Attributes:
        W1 (numpy.ndarray): The weights matrix for the hidden layer.
        b1 (numpy.ndarray): The bias vector for the hidden layer.
        A1 (float): The activated output for the hidden layer.
        W2 (numpy.ndarray): The weights matrix for the output layer.
        b2 (float): The bias for the output layer.
        A2 (float): The activated output for the output layer (prediction).
    """

    def __init__(self, nx, nodes):
        """Initialize a new NeuralNetwork instance.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0

    @property
    def W1(self):
        """Get the weights matrix for the hidden layer.

        Returns:
            numpy.ndarray: The weights matrix W1.
        """
        return self.__W1

    @property
    def b1(self):
        """Get the bias vector for the hidden layer.

        Returns:
            numpy.ndarray: The bias vector b1.
        """
        return self.__b1

    @property
    def A1(self):
        """Get the activated output for the hidden layer.

        Returns:
            float: The activated output A1.
        """
        return self.__A1

    @property
    def W2(self):
        """Get the weights matrix for the output layer.

        Returns:
            numpy.ndarray: The weights matrix W2.
        """
        return self.__W2

    @property
    def b2(self):
        """Get the bias for the output layer.

        Returns:
            float: The bias value b2.
        """
        return self.__b2

    @property
    def A2(self):
        """Get the activated output for the output layer.

        Returns:
            float: The activated output A2.
        """
        return self.__A2

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network.

        Updates the private attributes __A1 and __A2 using the sigmoid
        activation function for both the hidden and output layers.

        Args:
            X (numpy.ndarray): The input data with shape (nx, m).
                nx is the number of input features.
                m is the number of examples.

        Returns:
            tuple: (A1, A2)
                A1 (numpy.ndarray): The activated output of the hidden layer.
                A2 (numpy.ndarray): The activated output of the output layer.
        """
        temp = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-temp))
        temp = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-temp))
        return self.__A1, self.__A2

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
        preds = (self.__A2 >= 0.5).astype(int)
        cost = self.cost(Y, self.__A2)
        return (preds, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one iteration of gradient descent on the neural network.

        Updates the private attributes __W1, __b1, __W2, and __b2.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels for the input data, shape (1, m).
            A1 (numpy.ndarray): Activated output of the hidden layer.
            A2 (numpy.ndarray): Activated output of the output layer.
            alpha (float): The learning rate.
        """
        m = X.shape[1]
        temp2 = A2 - Y
        dW2 = (1 / m) * np.dot(temp2, A1.T)
        db2 = (1 / m) * np.sum(temp2)
        temp1 = np.dot(self.__W2.T, temp2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.dot(temp1, X.T)
        db1 = (1 / m) * np.sum(temp1, axis=1, keepdims=True)

        self.__W2 -= (alpha * dW2)
        self.__b2 -= (alpha * db2)
        self.__W1 -= (alpha * dW1)
        self.__b1 -= (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the neural network.

        Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels for the input data, shape (1, m).
            iterations (int): The number of iterations to train over.
            alpha (float): The learning rate.

        Raises:
            TypeError: If iterations is not an integer.
            ValueError: If iterations is not a positive integer.
            TypeError: If alpha is not a float.
            ValueError: If alpha is not positive.

        Returns:
            tuple: (prediction, cost) after the final iteration of training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if not alpha > 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
