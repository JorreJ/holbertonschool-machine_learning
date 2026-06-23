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
