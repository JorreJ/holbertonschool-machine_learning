#!/usr/bin/env python3
"""Module to perform batch normalization on a neural network layer."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize the activations of a layer using batch normalization.

    Args:
        Z (numpy.ndarray): The intermediate activations to be normalized.
        gamma (numpy.ndarray): The scale factor for the normalized value.
        beta (numpy.ndarray): The offset value for the normalized value.
        epsilon (float): A small constant to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized and scaled activations.
    """
    c_mean = np.mean(Z, axis=0)
    c_var = np.var(Z, axis=0)
    norm_data = (Z - c_mean) / np.sqrt(c_var + epsilon)
    return (gamma * norm_data) + beta
