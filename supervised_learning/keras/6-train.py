#!/usr/bin/env python3
"""Module to train a Keras model using specific data and labels."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent.

    Args:
        network (K.Model): The model to train.
        data (numpy.ndarray): The input data of shape (m, nx).
        labels (numpy.ndarray): The labels of shape (m, classes).
        batch_size (int): The size of the batch used for mini-batch
            gradient descent.
        epochs (int): The number of passes through the entire dataset.
        validation_data (tuple, optional): Data to validate the model with,
            as a tuple (val_data, val_labels). Defaults to None.
        early_stopping (bool): Whether to use early stopping. Only occurs
            if validation_data exists.
        patience (int): The patience for early stopping.
        verbose (bool): Whether or not to print training information.
        shuffle (bool): Whether or not to shuffle the training data.

    Returns:
        K.callbacks.History: The History object generated during training.
    """
    callbacks = []
    if early_stopping is True and validation_data:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        callbacks=callbacks,
        shuffle=shuffle
    )
