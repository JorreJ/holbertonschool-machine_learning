#!/usr/bin/env python3
"""Module that evaluates a trained Q-learning agent in an environment."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play an episode using a trained Q-table and render the environment.

    Args:
        env (gym.Env): The Gymnasium environment instance.
        Q (numpy.ndarray): 2D array representing the trained Q-table of shape
            (states, actions).
        max_steps (int, optional): Maximum number of steps to take in the
            episode. Defaults to 100.

    Returns:
        tuple:
            - total_rewards (float): Total reward accumulated during the
              episode.
            - rendered_outputs (list): List of rendered environment frames or
              strings for each step of the episode.
    """
    total_rewards = 0
    rendered_outputs = []
    state = env.reset()[0]
    rendered_outputs.append(env.render())
    for i in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        rendered_outputs.append(env.render())
        state = new_state
        if terminated or truncated:
            break
    return total_rewards, rendered_outputs
