#!/usr/bin/env python3
"""Module that performs SARSA(lambda) learning in Reinforcement Learning."""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform the SARSA(lambda) algorithm for action-value estimation.

    Args:
        env (gymnasium.Env): The environment instance.
        Q (numpy.ndarray): Array of shape (s, a) containing the Q-table.
        lambtha (float): The eligibility trace factor.
        episodes (int, optional): Total number of episodes to train.
            Defaults to 5000.
        max_steps (int, optional): Maximum number of steps per episode.
            Defaults to 100.
        alpha (float, optional): Learning rate. Defaults to 0.1.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        epsilon (float, optional): Initial exploration rate for
            epsilon-greedy policy. Defaults to 1.
        min_epsilon (float, optional): Minimum bound for epsilon decay.
            Defaults to 0.1.
        epsilon_decay (float, optional): Decay rate for epsilon per episode.
            Defaults to 0.05.

    Returns:
        numpy.ndarray: The updated Q-table Q.
    """
    initial_epsilon = epsilon
    for episode in range(episodes):
        state, info = env.reset()

        if np.random.uniform(0, 1) > epsilon:
            action = np.argmax(Q[state, :])
        else:
            action = np.random.randint(0, Q.shape[1])

        eligibility_traces = np.zeros_like(Q)
        for _ in range(max_steps):
            new_state, reward, terminated, truncated, info = env.step(action)
            if np.random.uniform(0, 1) > epsilon:
                new_action = np.argmax(Q[new_state, :])
            else:
                new_action = np.random.randint(0, Q.shape[1])
            delta = (reward + gamma * Q[new_state, new_action] -
                     Q[state, action])
            eligibility_traces[state, action] += 1
            Q += alpha * delta * eligibility_traces
            eligibility_traces *= lambtha * gamma
            state = new_state
            action = new_action
            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
