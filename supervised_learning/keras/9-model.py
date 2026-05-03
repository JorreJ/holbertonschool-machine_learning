#!/usr/bin/env python3
"""Module to save and load Keras models."""

import tensorflow.keras as K


def save_model(network, filename):
    """Save an entire model to a specific file.

    Args:
        network (K.Model): The model to save.
        filename (str): The path of the file the model should be saved to.

    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """Load an entire model from a specific file.

    Args:
        filename (str): The path of the file the model should be loaded from.

    Returns:
        K.Model: The loaded model.
    """
    return K.models.load_model(filename)
