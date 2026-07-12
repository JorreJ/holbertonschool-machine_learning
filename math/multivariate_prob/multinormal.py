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
