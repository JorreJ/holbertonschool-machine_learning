#!/usr/bin/env python3

"""This module plots the decay of Carbon-14 using a logarithmic scale."""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plot the exponential decay of Carbon-14.

    This function computes the fraction of remaining Carbon-14 over time
    using the exponential decay formula. The half-life of Carbon-14
    is used to determine the decay rate.

    The resulting graph uses a logarithmic scale on the y-axis to better
    visualize the decay over time.

    Returns:
        None
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y)
    plt.yscale('log')
    plt.xlim(0, 28650)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction remaining')
    plt.title('Exponential decay of C-14')
    plt.show()
