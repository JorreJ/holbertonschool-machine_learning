#!/usr/bin/env python3
"""Module to create mini-batches from a dataset."""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create mini-batches for mini-batch gradient descent.

    Args:
        X (numpy.ndarray): The input data of shape (m, nx) to batch.
        Y (numpy.ndarray): The labels of shape (m, ny) to batch.
        batch_size (int): The number of data points in each batch.

    Returns:
        list: A list of tuples (X_batch, Y_batch) containing the mini-batches.
    """
    new_X, new_Y = shuffle_data(X, Y)
    new_matrix = []
    for i in range(0, len(X), batch_size):
        new_tuple = (new_X[i:i + batch_size], new_Y[i:i + batch_size])
        new_matrix.append(new_tuple)
    return new_matrix
