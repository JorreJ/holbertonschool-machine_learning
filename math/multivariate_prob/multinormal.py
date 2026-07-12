#!/usr/bin/env python3
"""Module that defines the MultiNormal distribution class."""

import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initialize the Multivariate Normal distribution.

        Args:
            data (numpy.ndarray): A 2D array where each column represents
                a data point.

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If data contains fewer than 2 data points (columns).
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        data_center = data - self.mean
        self.cov = np.dot(data_center, data_center.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """Calculate the PDF value for a given data point.

        Args:
            x (numpy.ndarray): A 2D array of shape (d, 1) representing
                the data point.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x does not have the correct shape (d, 1).

        Returns:
            float: The PDF value at the data point x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({d}, 1)")

        term = (2 * np.pi) ** d
        det = np.linalg.det(self.cov)

        centered = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        exp = -0.5 * (centered.T @ inv_cov @ centered)

        return ((1 / np.sqrt(term * det)) * np.exp(exp)).item()
