#!/usr/bin/env python3
"""Module to create a momentum optimizer using TensorFlow."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Create a training operation using gradient descent with momentum.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        tf.keras.optimizers.Optimizer: A Keras optimizer instance configured
            with SGD and momentum.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
