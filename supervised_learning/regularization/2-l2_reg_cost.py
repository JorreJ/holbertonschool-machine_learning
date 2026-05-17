#!/usr/bin/env python3
"""Module to calculate the total cost with L2 regularization in Keras."""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculate the total cost of a Keras model including L2 regularization.

    Args:
        cost (tf.Tensor): A tensor containing the basic cost of the network
            without L2 regularization.
        model (tf.keras.Model): A Keras model that includes regularization
            losses.

    Returns:
        tf.Tensor: A tensor containing the total cost for each loss.
    """
    return tf.stack([cost + layer for layer in model.losses])
