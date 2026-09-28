#!/usr/bin/env python3
"""Module to evaluate a trained Keras model on a dataset."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test a Keras model on a given dataset.

    Args:
        network (K.Model): The Keras model to evaluate.
        data (numpy.ndarray or tf.Tensor): The input data for evaluation.
        labels (numpy.ndarray or tf.Tensor): The ground truth labels for data.
        verbose (bool, optional): Whether to print progress logs during
            evaluation. Defaults to True.

    Returns:
        list or float: The loss and accuracy (or metrics) of the model on the
        input data.
    """
    return network.evaluate(data, labels, verbose=verbose)
