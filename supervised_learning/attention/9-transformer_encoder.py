#!/usr/bin/env python3
"""Encoder for a Transformer."""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Defines a Transformer encoder."""

    def __init__(self, N, dm, h, hidden,
                 input_vocab, max_seq_len, drop_rate=0.1):
        """Initialize the Encoder layer.

        Args:
            N (int): Number of blocks in the encoder.
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the network.
            input_vocab (int): Size of the input vocabulary.
            max_seq_len (int): Maximum sequence length for positional encoding.
            drop_rate (float, optional): Dropout rate to be applied.
                Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_vocab, dm
        )
        self.positional_encoding = positional_encoding(
            max_seq_len, dm
        )
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate)
            for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Pass the input sequence through the encoder.

        Args:
            x (tf.Tensor): Tensor of shape (batch, input_seq_len)
                containing the input tokens.
            training (bool): Boolean indicating whether the model is in
                training mode.
            mask (tf.Tensor): Mask tensor to be applied for
                multi-head attention.

        Returns:
            tf.Tensor: Tensor of shape (batch, input_seq_len, dm)
            containing the encoder embeddings.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training, mask)
        return x
