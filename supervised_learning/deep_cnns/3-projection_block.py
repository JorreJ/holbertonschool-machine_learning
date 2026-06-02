#!/usr/bin/env python3
"""Module to build a projection block for a ResNet architecture."""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Build a projection block for a residual network.

    Args:
        A_prev (keras.Tensor): The activated output from the previous layer.
        filters (tuple or list): A tuple/list of 3 integers containing:
            - F11: The number of filters in the first 1x1 convolution.
            - F3: The number of filters in the 3x3 convolution.
            - F12: The number of filters in the second 1x1 convolution.
        s (int): The stride to be used for the first convolution in both the
            main path and the shortcut path. Defaults to 2.

    Returns:
        keras.Tensor: The activated output of the projection block.
    """
    F11, F3, F12 = filters

    # --- MAIN PATH ---

    A_main = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)
    A_main = K.layers.BatchNormalization(axis=3)(A_main)
    A_main = K.layers.Activation('relu')(A_main)

    A_main = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_main)
    A_main = K.layers.BatchNormalization(axis=3)(A_main)
    A_main = K.layers.Activation('relu')(A_main)

    A_main = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_main)
    A_main = K.layers.BatchNormalization(axis=3)(A_main)

    # --- SHORTCUT PATH ---

    A_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)
    A_shortcut = K.layers.BatchNormalization(axis=3)(A_shortcut)

    # --- FUSION ---

    A = K.layers.Add()([A_main, A_shortcut])
    A = K.layers.Activation('relu')(A)

    return A
