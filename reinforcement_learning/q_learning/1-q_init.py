#!/usr/bin/env python3
"""Module that initializes the Q-table for reinforcement learning."""

import numpy as np


def q_init(env):
    """Initialize a Q-table with zeros for a given Gymnasium environment.

    Args:
        env (gym.Env): The Gymnasium environment instance.

    Returns:
        numpy.ndarray: A 2D array of zeros with shape (states, actions).
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
