#!/usr/bin/env python3
"""Module for Bahdanau Self-Attention mechanism in TensorFlow."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculate Bahdanau attention for machine translation."""

    def __init__(self, units):
        """Initialize SelfAttention layer.

        Args:
            units (int): The number of hidden units in the dense layers.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """Perform the forward pass for attention mechanism.

        Args:
            s_prev (tf.Tensor): Previous decoder hidden state of shape
                (batch, units).
            hidden_states (tf.Tensor): Encoder hidden states of shape
                (batch, input_seq_len, units).

        Returns:
            tuple:
                - context (tf.Tensor): Context vector of shape (batch, units).
                - weights (tf.Tensor): Attention weights of shape
                  (batch, input_seq_len, 1).
        """
        score = self.V(
            tf.math.tanh(
                self.W(s_prev[:, tf.newaxis, :]) + self.U(hidden_states)
            )
        )
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
