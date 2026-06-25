#!/usr/bin/env python3
"""Module that defines the Binomial distribution class."""


class Binomial:
    """Class that represents a Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize the Binomial distribution.

        Args:
            data (list, optional): A list of data points to estimate the
                distribution's n and p. Defaults to None.
            n (int, optional): The number of trials. Defaults to 1.
            p (float, optional): The probability of success. Defaults to 0.5.

        Raises:
            TypeError: If data is provided but is not a list.
            ValueError: If n is less than 1 when data is None.
            ValueError: If p is not between 0 and 1 (exclusive) when data
                is None.
            ValueError: If data contains fewer than 2 values.
        """
        if data is None:
            if n < 1:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            base_p = 1 - (variance / mean)
            self.n = int(round(mean / base_p))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculate the value of the PMF for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k successes.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorielle(x):
            fact = 1
            for i in range(1, x + 1):
                fact *= i
            return fact

        n_fact = factorielle(self.n)
        k_fact = factorielle(k)
        n_k_fact = factorielle(self.n - k)

        bi_co = n_fact / (k_fact * n_k_fact)

        return bi_co * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculate the value of the CDF for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k successes.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        total_prob = 0.0
        for i in range(k + 1):
            total_prob += self.pmf(i)

        return float(total_prob)
