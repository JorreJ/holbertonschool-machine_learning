#!/usr/bin/env python3
"""Masking utility functions for Transformer model inputs and targets."""

import tensorflow as tf


def create_padding_mask(input):
    """Create a padding mask for a batch of sequences.

    Args:
        input (tf.Tensor): Tensor of shape (batch_size, seq_len) where 0
            represents padding tokens.

    Returns:
        tf.Tensor: Padding mask tensor of shape
        (batch_size, 1, 1, seq_len) with 1.0 for padding positions and
        0.0 otherwise.
    """
    mask = tf.cast(tf.math.equal(input, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """Create a look-ahead mask to prevent attention to future tokens.

    Args:
        size (int): The sequence length (square matrix dimension).

    Returns:
        tf.Tensor: Upper triangular mask tensor of shape (size, size) with
        1.0 for masked positions and 0.0 for unmasked positions.
    """
    mask = 1 - tf.linalg.band_part(
        tf.ones((size, size)), -1, 0
    )
    return mask


def create_masks(inputs, target):
    """Create encoder, combined decoder, and decoder padding masks.

    Args:
        inputs (tf.Tensor): Source input sequence tensor of shape
            (batch_size, seq_len_in).
        target (tf.Tensor): Target sequence tensor of shape
            (batch_size, seq_len_out).

    Returns:
        tuple:
            - encoder_mask (tf.Tensor): Padding mask for the encoder.
            - combined_mask (tf.Tensor): Mask combining look-ahead and
              target padding masks for the first decoder layer.
            - decoder_mask (tf.Tensor): Padding mask for the second decoder
              layer (cross-attention).
    """
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)
    target_padding_mask = create_padding_mask(target)
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)
    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)
    return encoder_mask, combined_mask, decoder_mask
