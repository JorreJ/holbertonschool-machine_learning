#!/usr/bin/env python3
"""Module for RNN Encoder implementation using TensorFlow."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder layer for sequence-to-sequence models."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNNEncoder layer.

        Args:
            vocab (int): The size of the input vocabulary.
            embedding (int): The dimensionality of the embedding vector.
            units (int): The number of hidden units in the GRU cell.
            batch (int): The batch size.
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )

    def initialize_hidden_state(self):
        """Initialize the hidden state of the GRU to zeros.

        Returns:
            tf.Tensor: A tensor of shape (batch, units) filled with zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Perform the forward pass of the encoder.

        Args:
            x (tf.Tensor): Tensor of shape (batch, input_seq_len) with
                input token IDs.
            initial (tf.Tensor): Tensor of shape (batch, units) representing
                the initial hidden state.

        Returns:
            tuple:
                - outputs (tf.Tensor): GRU outputs tensor of shape
                  (batch, input_seq_len, units).
                - hidden (tf.Tensor): Last GRU hidden state tensor of shape
                  (batch, units).
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(inputs=x, initial_state=initial)
        return outputs, hidden
