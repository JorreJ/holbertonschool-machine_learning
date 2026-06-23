#!/usr/bin/env python3
"""Module that defines the Poisson distribution class."""


class Poisson:
    """Class that represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1):
        """Initialize the Poisson distribution.

        Args:
            data (list, optional): A list of data points to estimate the
                distribution's lambtha. Defaults to None.
            lambtha (float, optional): The expected number of occurrences
                (lambda) during a given time interval. Defaults to 1.

        Raises:
            TypeError: If data is provided but is not a list.
            ValueError: If lambtha is less than or equal to 0.
            ValueError: If data contains fewer than 2 values.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha: float = lambtha
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha: float = sum(data) / len(data)

    def pmf(self, k):
        """Calculate the value of the PMF for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k successes.
        """
        k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        exp = e ** (-self.lambtha)
        lambtha_pow = self.lambtha ** k
        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i

        return (lambtha_pow * exp) / k_fact

    def cdf(self, k):
        """Calculate the value of the CDF for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k successes.
        """
        k = int(k)
        if k < 0:
            return 0

        total_prob = 0.0
        for i in range(k + 1):
            total_prob += self.pmf(i)

        return total_prob
