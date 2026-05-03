#!/usr/bin/env python3
"""Module to save and load Keras model weights."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save the weights of a model to a specific file.

    Args:
        network (K.Model): The model whose weights should be saved.
        filename (str): The path of the file the weights should be saved to.
        save_format (str): The format in which the weights should be saved.
            Defaults to 'keras'.

    Returns:
        None
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Load the weights of a model from a specific file.

    Args:
        network (K.Model): The model to which the weights should be loaded.
        filename (str): The path of the file the weights should be loaded from.

    Returns:
        None
    """
    network.load_weights(filepath=filename)
    return None
