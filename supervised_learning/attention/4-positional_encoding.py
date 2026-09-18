#!/usr/bin/env python3
"""Module that calculates positional encoding for Transformers."""

import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Calculate the positional encoding for a sequence.

    Args:
        max_seq_len (int): The maximum sequence length.
        dm (int): The model depth / dimensionality of the embeddings.

    Returns:
        numpy.ndarray: Positional encoding matrix of shape (max_seq_len, dm).
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / dm)
    pe = np.zeros((max_seq_len, dm))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe
