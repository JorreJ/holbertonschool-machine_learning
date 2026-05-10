#!/usr/bin/env python3
"""Module to create an Adam optimizer using TensorFlow."""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Create a training operation using the Adam optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): Exponential decay rate of the first moment estimates.
        beta2 (float): Exponential decay rate of the second moment estimates.
        epsilon (float): A small constant for numerical stability.

    Returns:
        tf.keras.optimizers.Optimizer: An Adam optimizer instance.
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
