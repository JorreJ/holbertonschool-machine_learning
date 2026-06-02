#!/usr/bin/env python3
"""Module to build an identity block for a ResNet architecture."""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Build an identity block for a residual network.

    Args:
        A_prev (keras.Tensor): The activated output from the previous layer.
        filters (tuple or list): A tuple/list of 3 integers containing:
            - F11: The number of filters in the first 1x1 convolution.
            - F3: The number of filters in the 3x3 convolution.
            - F12: The number of filters in the second 1x1 convolution.

    Returns:
        keras.Tensor: The activated output of the identity block.
    """
    F11, F3, F12 = filters
    A = A_prev

    A_prev = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.Activation('relu')(A_prev)

    A_prev = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.Activation('relu')(A_prev)

    A_prev = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)

    A_prev = K.layers.Add()([A_prev, A])

    A_prev = K.layers.Activation('relu')(A_prev)

    return A_prev
