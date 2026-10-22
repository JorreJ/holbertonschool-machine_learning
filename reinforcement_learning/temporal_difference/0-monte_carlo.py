#!/usr/bin/env python3
"""Performs Monte Carlo control/evaluation in Reinforcement Learning."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Perform the first-visit Monte Carlo algorithm for value estimation.

    Args:
        env (gymnasium.Env): The environment instance.
        V (numpy.ndarray): Array of shape (s,) containing the value function.
        policy (callable): Function that takes a state and returns an action.
        episodes (int, optional): Total number of episodes to train.
            Defaults to 5000.
        max_steps (int, optional): Maximum number of steps per episode.
            Defaults to 100.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.

    Returns:
        numpy.ndarray: The updated value function V.
    """
    for episode in range(episodes):
        state, info = env.reset()
        data = []
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            data.append((state, reward))
            state = next_state
            if terminated or truncated:
                break
        G = 0
        data = np.array(data, dtype=int)
        for state, reward in reversed(data):
            G = reward + gamma * G
            if state not in data[:episode, 0]:
                V[state] = V[state] + alpha * (G - V[state])
    return V
