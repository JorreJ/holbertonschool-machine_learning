#!/usr/bin/env python3

"""This module generates and displays a histogram of student grades."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plot a histogram of student grades.

    This function generates a sample of student grades using a normal
    distribution, then displays their frequency distribution using a histogram.

    The grades are grouped into bins of size 10, ranging from 0 to 100.

    Returns:
        None
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
