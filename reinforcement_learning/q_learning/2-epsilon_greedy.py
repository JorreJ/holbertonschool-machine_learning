#!/usr/bin/env python3
"""Module that implements the epsilon-greedy action selection strategy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select an action using the epsilon-greedy policy.

    Args:
        Q (numpy.ndarray): A 2D array containing the Q-table of shape
            (states, actions).
        state (int): The current state index.
        epsilon (float): The probability of choosing a random exploration
            action.

    Returns:
        int: The index of the selected action.
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        max_ids = np.where(Q[state, :] == max(Q[state, :]))[0]
        action = np.random.choice(max_ids)
    return action
