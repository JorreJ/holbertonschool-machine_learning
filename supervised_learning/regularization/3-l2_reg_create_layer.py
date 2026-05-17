#!/usr/bin/env python3
"""Module to create a dense layer with L2 regularization in TensorFlow."""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create a dense layer with L2 regularization.

    Args:
        prev (tf.Tensor): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation (function): The activation function to be used on the
            output of the layer.
        lambtha (float): The L2 regularization parameter.

    Returns:
        tf.Tensor: The output of the newly created layer.
    """
    return tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha),
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg"
        )
    )(prev)
