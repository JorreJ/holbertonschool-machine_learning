#!/usr/bin/env python3
"""Module that defines the BayesianOptimization class."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Represent a Bayesian Optimization model."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initialize the Bayesian Optimization model.

        Args:
            f (function): The black-box function to be optimized.
            X_init (numpy.ndarray): A 2D array of shape (t, 1) representing
                the inputs already sampled.
            Y_init (numpy.ndarray): A 2D array of shape (t, 1) representing
                the outputs of the black-box function for each input in X_init.
            bounds (tuple): A tuple of (min, max) representing the bounds
                of the space in which to look for the optimal point.
            ac_samples (int): The number of samples that should be analyzed
                during acquisition.
            l (float or int): The length-scale parameter for the kernel.
                Defaults to 1.
            sigma_f (float or int): Signal variance parameter for the kernel.
                Defaults to 1.
            xsi (float): The exploration-exploitation factor for acquisition.
                Defaults to 0.01.
            minimize (bool): A boolean determining whether optimization should
                be performed for minimization (True) or maximization (False).
                Defaults to True.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculate the next best sample location using Expected Improvement.

        Returns:
            tuple: A tuple (X_next, EI) where:
                - X_next (numpy.ndarray): A 1D array of shape (1,)
                  representing the next best sample point.
                - EI (numpy.ndarray): A 1D array of shape (ac_samples,)
                  containing the Expected Improvement for each point in X_s.
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_best = np.min(self.gp.Y)
            imp = Y_best - mu - self.xsi
        else:
            Y_best = np.max(self.gp.Y)
            imp = mu - Y_best - self.xsi

        Z = imp / sigma

        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0] = 0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Optimize the black-box function using Bayesian Optimization.

        Args:
            iterations (int): The maximum number of iterations to perform.
                Defaults to 100.

        Returns:
            tuple: A tuple (X_opt, Y_opt) where:
                - X_opt (numpy.ndarray): A 1D array of shape (1,) representing
                  the optimal point.
                - Y_opt (numpy.ndarray): A 1D array of shape (1,) representing
                  the function value at X_opt.
        """
        for i in range(iterations):
            X_next, EI = self.acquisition()

            if np.any(np.all(self.gp.X == X_next, axis=1)):
                break

            Y_next = self.f(X_next)

            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        return self.gp.X[idx], self.gp.Y[idx]
