#!/usr/bin/env python3

"""This module compares the exponential decay of two radioactive elements."""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plot the exponential decay of two radioactive elements.

    This function computes and displays the decay curves of Carbon-14
    and Radium-226 using their respective half-lives. The results are
    plotted on the same graph for comparison.

    The decay is modeled using an exponential function based on the
    half-life of each element.

    Returns:
        None
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600

    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')
    plt.legend()
    plt.show()
