#!/usr/bin/env python3
"""Module that defines a multi-head attention layer using TensorFlow."""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Perform multi-head attention on input query, key, and value tensors."""

    def __init__(self, dm, h):
        """Initialize the MultiHeadAttention layer.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()

        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension of the tensor into (h, depth).

        Args:
            x (tf.Tensor): Tensor of shape (batch_size, seq_len, dm).
            batch_size (int or tf.Tensor): The batch size of the input.

        Returns:
            tf.Tensor: Transposed tensor of shape
            (batch_size, h, seq_len, depth).
        """
        x = tf.reshape(
            x, (batch_size, -1, self.h, self.depth)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask=None):
        """Perform multi-head attention evaluation.

        Args:
            Q (tf.Tensor): Query tensor of shape (batch_size, seq_len_q, dk).
            K (tf.Tensor): Key tensor of shape (batch_size, seq_len_v, dk).
            V (tf.Tensor): Value tensor of shape (batch_size, seq_len_v, dv).
            mask (tf.Tensor, optional): Mask tensor broadcastable to shape
                (batch_size, ..., seq_len_q, seq_len_v). Defaults to None.

        Returns:
            tuple:
                - output (tf.Tensor): Layer output tensor of shape
                  (batch_size, seq_len_q, dm).
                - weights (tf.Tensor): Attention weights tensor of shape
                  (batch_size, h, seq_len_q, seq_len_v).
        """
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            output,
            (batch_size, -1, self.dm)
        )
        output = self.linear(concat_attention)
        return output, weights
