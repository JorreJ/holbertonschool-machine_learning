#!/usr/bin/env python3
"""Module to create a learning rate decay schedule using TensorFlow."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Create a learning rate decay schedule with a staircase staircase.

    Args:
        alpha (float): The initial learning rate.
        decay_rate (float): The weight used to determine the rate at which
            the learning rate will decay.
        decay_step (int): The number of passes of gradient descent that should
            occur before the learning rate is decayed further.

    Returns:
        tf.keras.optimizers.schedules.InverseTimeDecay: A learning rate
            schedule object.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
