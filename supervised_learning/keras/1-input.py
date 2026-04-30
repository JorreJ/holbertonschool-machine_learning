#!/usr/bin/env python3
"""Module to build a Keras model using the functional API."""

import tensorflow.keras as K


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
        K.Model: The Keras model instance.
    """
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=reg
        )(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
