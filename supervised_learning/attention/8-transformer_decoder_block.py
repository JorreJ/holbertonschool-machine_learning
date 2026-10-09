#!/usr/bin/env python3
"""Decoder block for a Transformer."""

import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Defines a Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the DecoderBlock layer.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the network.
            drop_rate (float, optional): Dropout rate to be applied.
                Defaults to 0.1.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
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
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """Pass the input through the decoder block layer.

        Args:
            x (tf.Tensor): Tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder block.
            encoder_output (tf.Tensor): Tensor of shape
                (batch, input_seq_len, dm) containing
                the output of the encoder.
            training (bool): Boolean indicating whether the model is in
                training mode.
            look_ahead_mask (tf.Tensor): Mask tensor to be applied to the first
                multi-head attention layer.
            padding_mask (tf.Tensor): Mask tensor to be applied to the second
                multi-head attention layer.

        Returns:
            tf.Tensor: Tensor of shape (batch, target_seq_len, dm)
            containing the block output.
        """
        attention1, _ = self.mha1(
            x, x, x, look_ahead_mask
        )
        attention1 = self.dropout1(
            attention1, training=training
        )
        out1 = self.layernorm1(x + attention1)
        attention2, _ = self.mha2(
            out1, encoder_output, encoder_output,
            padding_mask
        )
        attention2 = self.dropout2(
            attention2, training=training
        )
        out2 = self.layernorm2(out1 + attention2)
        hidden = self.dense_hidden(out2)
        output = self.dense_output(hidden)
        output = self.dropout3(output, training=training)
        return self.layernorm3(out2 + output)
