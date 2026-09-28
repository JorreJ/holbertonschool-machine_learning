#!/usr/bin/env python3
"""Module that trains a Q-learning agent on a Gymnasium environment."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
    env,
    Q,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05
):
    """Perform Q-learning training on a given Gymnasium environment.

    Args:
        env (gym.Env): The Gymnasium environment instance.
        Q (numpy.ndarray): 2D array representing the Q-table of shape
            (states, actions).
        episodes (int, optional): Total number of training episodes.
            Defaults to 5000.
        max_steps (int, optional): Maximum steps per episode.
            Defaults to 100.
        alpha (float, optional): Learning rate parameter.
            Defaults to 0.1.
        gamma (float, optional): Discount factor parameter.
            Defaults to 0.99.
        epsilon (float, optional): Initial exploration rate parameter.
            Defaults to 1.
        min_epsilon (float, optional): Minimum exploration rate limit.
            Defaults to 0.1.
        epsilon_decay (float, optional): Decay amount applied to epsilon
            after each episode. Defaults to 0.05.

    Returns:
        tuple:
            - Q (numpy.ndarray): The updated Q-table.
            - total_rewards (list): List containing the total cumulative reward
              obtained in each episode.
    """
    total_rewards = []
    for i in range(episodes):
        state = env.reset()[0]
        step = 0
        done = False
        episode_reward = 0
        while not done and step < max_steps:
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)
            if env.unwrapped.desc.flat[new_state] == b'H':
                reward = -1
            done = terminated or truncated
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action]
            )
            episode_reward += reward
            state = new_state
            step += 1
        total_rewards.append(episode_reward)
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    return Q, total_rewards
