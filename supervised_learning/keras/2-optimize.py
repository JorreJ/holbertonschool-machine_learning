#!/usr/bin/env python3
"""Module to set up the optimization for a Keras model."""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Set up the Adam optimization for a keras model.

    Args:
        network (K.Model): The model to optimize.
        alpha (float): The learning rate.
        beta1 (float): The exponential decay rate for the first
            moment estimates.
        beta2 (float): The exponential decay rate for the second
            moment estimates.

    Returns:
        None
    """
    network.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=alpha,
            beta_1=beta1,
            beta_2=beta2
        ),
        loss='categorical_crossentropy'
    )
