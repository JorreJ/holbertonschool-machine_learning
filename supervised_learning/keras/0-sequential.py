#!/usr/bin/env python3
"""Module to build a deep learning model using TensorFlow Keras."""

import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network with L2 regularization and dropout.

    Args:
        nx (int): The number of input features.
        layers (list): A list containing the number of nodes in each layer.
        activations (list): A list containing the activation functions for
            each layer.
        lambtha (float): The L2 regularization parameter.
        keep_prob (float): The probability that a node will be kept.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = tf.keras.Sequential()
    for x in range(len(layers)):
        kwargs = {
            'units': layers[x],
            'activation': activations[x],
            'kernel_regularizer': tf.keras.regularizers.l2(lambtha)
        }
        if x == 0:
            kwargs['input_shape'] = (nx,)

        model.add(tf.keras.layers.Dense(**kwargs))
        if x < len(layers) - 1:
            model.add(tf.keras.layers.Dropout(1 - keep_prob))

    return model
