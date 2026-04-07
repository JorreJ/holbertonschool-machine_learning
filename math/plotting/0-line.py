#!/usr/bin/env python3

"""This module provides a function to plot a simple cubic line graph."""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot a cubic function.

    This function generates values from 0 to 10 and computes their cubes.
    It then plots the resulting values as a red line on a graph.

    The x-axis represents the input values (0 to 10),
    and the y-axis represents their cubes.

    Returns:
        None
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, 'r-')
    plt.xlim(0, 10)
    plt.show()
