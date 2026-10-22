#!/usr/bin/env python3
"""Module that performs TD(lambda) learning in Reinforcement Learning."""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Perform the TD(lambda) algorithm for value estimation.

    Args:
        env (gymnasium.Env): The environment instance.
        V (numpy.ndarray): Array of shape (s,) containing the value function.
        policy (callable): Function that takes a state and returns an action.
        lambtha (float): The eligibility trace factor.
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
        eligibility = np.zeros_like(V)
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            state = next_state
            if terminated or truncated:
                break
    return V
