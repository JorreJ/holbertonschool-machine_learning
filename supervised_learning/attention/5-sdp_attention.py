#!/usr/bin/env python3
"""Module that calculates scaled dot product attention using TensorFlow."""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot product attention.

    Args:
        Q (tf.Tensor): Query tensor of shape (..., seq_len_q, dk).
        K (tf.Tensor): Key tensor of shape (..., seq_len_v, dk).
        V (tf.Tensor): Value tensor of shape (..., seq_len_v, dv).
        mask (tf.Tensor, optional): Tensor containing mask values
            broadcastable to (..., seq_len_q, seq_len_v). Defaults to None.

    Returns:
        tuple:
            - output (tf.Tensor): Tensor of shape (..., seq_len_q, dv)
              containing the scaled dot product attention output.
            - weights (tf.Tensor): Tensor of shape (..., seq_len_q, seq_len_v)
              containing the attention weights.
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], Q.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += mask * tf.cast(
            -1e9, scaled_attention_logits.dtype
        )
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights
