#!/usr/bin/env python3
"""Module to save and load Keras model configurations in JSON format."""

import tensorflow.keras as K


def save_config(network, filename):
    """Save a model's architecture configuration to a JSON file.

    Args:
        network (K.Model): The model whose architecture should be saved.
        filename (str): The path to the file where the JSON string will be
            written.

    Returns:
        None
    """
    with open(filename, "w") as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """Load a model architecture from a JSON file.

    Args:
        filename (str): The path to the JSON file containing the model
            architecture.

    Returns:
        K.Model: The loaded Keras model instance.
    """
    with open(filename, "r") as f:
        config = f.read()
    return K.models.model_from_json(config)
