#!/usr/bin/env python3
"""Module that defines the Normal distribution class."""


class Normal:
    """Class that represents a Normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the Normal distribution.

        Args:
            data (list, optional): A list of data points to estimate the
                distribution's mean and stddev. Defaults to None.
            mean (float, optional): The mean of the distribution.
                Defaults to 0.0.
            stddev (float, optional): The standard deviation of the
                distribution. Defaults to 1.0.

        Raises:
            TypeError: If data is provided but is not a list.
            ValueError: If stddev is less than or equal to 0 when data is None.
            ValueError: If data contains fewer than 2 values.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            squared_stddev = sum((x - self.mean) ** 2 for x in data)
            self.stddev = float((squared_stddev / len(data)) ** 0.5)

    def z_score(self, x):
        """Calculate the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: The x-value of z.
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """Calculate the value of the PDF for a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        pi = 3.1415926536
        e = 2.7182818285
        z = self.z_score(x)

        exp = -0.5 * (z ** 2)
        square_root = (2 * pi) ** 0.5

        return (1 / (self.stddev * square_root)) * (e ** exp)
