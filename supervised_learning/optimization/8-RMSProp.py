#!/usr/bin/env python3
"""Module to create an RMSProp optimizer using TensorFlow."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Create a training operation using the RMSProp optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The decay rate (rho) for the second moment.
        epsilon (float): A small constant to avoid division by zero.

    Returns:
        tf.keras.optimizers.Optimizer: An RMSProp optimizer instance.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
