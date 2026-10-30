#!/usr/bin/env python3
"""Module that trains a policy using Policy Gradient algorithm."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Train a policy using Monte Carlo Policy Gradient (REINFORCE).

    Args:
        env (gymnasium.Env): The environment instance.
        nb_episodes (int): Total number of episodes to train.
        alpha (float, optional): Learning rate. Defaults to 0.000045.
        gamma (float, optional): Discount factor. Defaults to 0.98.

    Returns:
        list: List containing the cumulative score (reward) achieved in
        each episode.
    """
    weight = np.random.rand(
        env.observation_space.shape[0], env.action_space.n
    )
    scores = []
    for episode in range(nb_episodes):
        state, info = env.reset()
        data = []
        score = 0
        done = False
        while not done:
            action, gradient = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, info = env.step(action)
            data.append((reward, gradient))
            score += reward
            state = next_state
            if terminated or truncated:
                done = True
        G = 0
        episode_data = []
        for reward, gradient in reversed(data):
            G = reward + gamma * G
            episode_data.insert(0, (G, gradient))
        for G, gradient in episode_data:
            weight += alpha * G * gradient
        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))
    return scores
