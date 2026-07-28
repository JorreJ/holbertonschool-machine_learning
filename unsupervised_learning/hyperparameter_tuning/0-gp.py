#!/usr/bin/env python3
"""Module that defines a Gaussian Process class."""

import numpy as np


class GaussianProcess:
    """Represent a 1D Gaussian Process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian Process.

        Args:
            X_init (numpy.ndarray): A 2D array of shape (m, 1) representing
                the inputs already sampled.
            Y_init (numpy.ndarray): A 2D array of shape (m, 1) representing
                the outputs for the inputs already sampled.
            l (float/int): The length-scale parameter for the kernel.
                Defaults to 1.
            sigma_f (float/int): The signal variance parameter for the kernel.
                Defaults to 1.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculate the Radial Basis Function (RBF) covariance kernel matrix.

        Args:
            X1 (numpy.ndarray): A 2D array of shape (m, 1) containing inputs.
            X2 (numpy.ndarray): A 2D array of shape (n, 1) containing inputs.

        Returns:
            numpy.ndarray: A covariance kernel matrix of shape (m, n).
        """
        sqdist = (X1 - X2.T) ** 2
        return self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))
