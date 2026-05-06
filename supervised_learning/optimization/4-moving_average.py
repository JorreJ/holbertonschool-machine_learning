#!/usr/bin/env python3
"""Module to calculate the exponentially weighted moving average of a list."""


def moving_average(data, beta):
    """Calculate an exponentially weighted moving average with bias correction.

    Args:
        data (list): A list of data points to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        list: A list containing the moving averages of data.
    """
    v = 0
    moving_averages = []
    for i, data_point in enumerate(data, start=1):
        v = (beta * v) + ((1 - beta) * data_point)
        v_corrected = v / (1 - (beta ** i))
        moving_averages.append(v_corrected)
    return moving_averages
