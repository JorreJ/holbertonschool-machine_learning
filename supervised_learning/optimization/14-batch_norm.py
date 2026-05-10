#!/usr/bin/env python3
"""Module to create a batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tf.Tensor): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation (function): The activation function to be used on the
            output of the layer.

    Returns:
        tf.Tensor: The activated output for the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )
    normalization = tf.keras.layers.BatchNormalization(epsilon=1e-7)
    x = layer(prev)
    x = normalization(x, training=True)
    return activation(x)
