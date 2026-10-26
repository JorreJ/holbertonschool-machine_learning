#!/usr/bin/env python3
"""Module that computes policy probabilities using softmax."""

import numpy as np


def policy(matrix, weight):
    """Compute the softmax policy probabilities for a set of states.

    Args:
        matrix (numpy.ndarray): 2D array of shape (n, d) containing the state
            features.
        weight (numpy.ndarray): 2D array of shape (d, a) containing the policy
            weights.

    Returns:
        numpy.ndarray: 2D array of shape (n, a) containing the policy
        probabilities for each state and action.
    """
    z = matrix @ weight
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Calculate the Monte Carlo policy gradient of a state and weight.

    Args:
        state (numpy.ndarray): 1D array of shape (d,) containing the state
            features.
        weight (numpy.ndarray): 2D array of shape (d, a) containing the policy
            weights.

    Returns:
        tuple:
            - action (int): The selected action index based on policy
              probabilities.
            - gradient (numpy.ndarray): 2D array of shape (d, a) containing
              the computed policy gradient.
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)
    action = np.random.choice([0, 1], p=probs[0])
    gradient = state.T @ (np.eye(2)[action] - probs)
    return action, gradient
