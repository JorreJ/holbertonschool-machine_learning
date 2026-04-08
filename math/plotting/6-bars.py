#!/usr/bin/env python3

"""This module generates stacked bar charts of fruit quantities per person."""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plot a stacked bar chart of fruit quantities.

    This function generates random quantities of four types of fruit
    (apples, bananas, oranges, peaches) for three individuals.
    The data is displayed as a stacked bar chart showing the total
    quantity of fruit per person.

    Returns:
        None
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    x = ['Farrah', 'Fred', 'Felicia']
    apple = fruit[0]
    banana = fruit[1]
    orange = fruit[2]
    peach = fruit[3]

    plt.figure(figsize=(6.4, 4.8))
    plt.ylim(0, 80)
    plt.bar(x, apple, color='r', width=0.5)
    plt.bar(x, banana, bottom=apple, color='yellow', width=0.5)
    plt.bar(x, orange, bottom=apple+banana, color='#ff8000', width=0.5)
    plt.bar(x, peach, bottom=apple+banana+orange, color='#ffe5b4', width=0.5)

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend(['apples', 'bananas', 'oranges', 'peaches'])
    plt.show()
