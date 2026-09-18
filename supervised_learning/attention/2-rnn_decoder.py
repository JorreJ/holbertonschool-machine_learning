#!/usr/bin/env python3
"""Module for RNN Decoder implementation with attention using TensorFlow."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder layer for sequence-to-sequence models with attention."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNNDecoder layer.

        Args:
            vocab (int): The size of the output vocabulary.
            embedding (int): The dimensionality of the embedding vector.
            units (int): The number of hidden units in the GRU cell.
            batch (int): The batch size.
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Perform the forward pass of the decoder.

        Args:
            x (tf.Tensor): Tensor of shape (batch, 1) containing the previous
                target token ID.
            s_prev (tf.Tensor): Tensor of shape (batch, units) containing the
                previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape
                (batch, input_seq_len, units) containing the encoder hidden
                states.

        Returns:
            tuple:
                - y (tf.Tensor): Output tensor of shape (batch, vocab) with
                  unnormalized predictions for the next token.
                - s (tf.Tensor): New decoder hidden state tensor of shape
                  (batch, units).
        """
        context, _ = self.attention(s_prev, hidden_states)
        context = context[:, tf.newaxis, :]
        x = self.embedding(x)
        inputs = tf.concat([context, x], axis=2)
        y, s = self.gru(inputs=inputs, initial_state=s_prev)
        y = self.F(tf.squeeze(y, axis=1))
        return y, s
