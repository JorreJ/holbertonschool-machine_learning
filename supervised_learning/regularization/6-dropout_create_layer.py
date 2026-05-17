#!/usr/bin/env python3
"""Module to create a layer with Dropout regularization in TensorFlow."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Create a neural network layer using dropout.

    Args:
        prev (tensor): Output of the previous layer.
        n (int): Number of nodes the new layer should contain.
        activation (callable): Activation function for the new layer.
        keep_prob (float): Probability that a node will be kept.
        training (bool): Indicating whether the model is in training mode.

    Returns:
        tensor: The output of the new layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode="fan_avg"
    )

    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )(prev)

    rate = 1 - keep_prob
    dropout_layer = tf.keras.layers.Dropout(rate=rate)(
        dense_layer,
        training=training
    )

    return dropout_layer
