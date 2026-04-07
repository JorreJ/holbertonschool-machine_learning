#!/usr/bin/env python3

"""This module and displays a scatter plot of height and weight data."""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Generate and display a scatter plot of height vs weight.

    This function creates a dataset of 2000 samples drawn from a
    multivariate normal distribution representing height and weight.
    The weight values are adjusted and plotted against height.

    The resulting scatter plot visualizes the relationship between
    height (in inches) and weight (in pounds).

    Returns:
        None
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]

    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, color='magenta')
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.show()
