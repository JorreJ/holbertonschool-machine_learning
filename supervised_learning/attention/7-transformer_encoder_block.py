#!/usr/bin/env python3
"""Module that defines an encoder block layer for Transformer models."""

import tensorflow as tf

MultiHeadAttention = __import__(
    '6-multihead_attention'
).MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Represent an encoder block for a Transformer network."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the EncoderBlock layer.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the network.
            drop_rate (float, optional): Dropout rate to be applied.
                Defaults to 0.1.
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden, activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Pass the input through the encoder block layer.

        Args:
            x (tf.Tensor): Tensor of shape (batch, input_seq_len, dm)
                containing the input to the encoder block.
            training (bool): Boolean indicating whether the model is in
                training mode.
            mask (tf.Tensor, optional): Mask tensor to be applied for
                multi-head attention. Defaults to None.

        Returns:
            tf.Tensor: Tensor of shape (batch, input_seq_len, dm)
            containing the block output.
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(
            attention, training=training
        )
        x1 = self.layernorm1(x + attention)
        hidden = self.dense_hidden(x1)
        output = self.dense_output(hidden)
        output = self.dropout2(
            output, training=training
        )
        output = self.layernorm2(x1 + output)
        return output
