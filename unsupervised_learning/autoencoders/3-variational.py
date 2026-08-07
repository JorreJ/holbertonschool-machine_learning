#!/usr/bin/env python3
"""Module that provides functions to build a Variational Autoencoder (VAE)."""

import tensorflow.keras as keras


def sampling(args):
    """Sample from a Gaussian distribution using the reparameterization trick.

    Args:
        args (tuple): A tuple (mean, log_var) containing:
            - mean (Tensor): The mean of the latent distribution.
            - log_var (Tensor): The log variance of the latent distribution.

    Returns:
        Tensor: A sampled tensor z from the latent distribution.
    """
    mean, log_var = args

    epsilon = keras.backend.random_normal(
        shape=keras.backend.shape(mean)
    )

    return mean + keras.backend.exp(log_var / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a Variational Autoencoder model, encoder, and decoder.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list containing the number of nodes for each
            hidden layer in the encoder, respectively.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        tuple: A tuple (encoder, decoder, auto) where:
            - encoder (keras.Model): The encoder model.
            - decoder (keras.Model): The decoder model.
            - auto (keras.Model): The full VAE model.
    """
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for unit in hidden_layers:
        x = keras.layers.Dense(
            units=unit,
            activation='relu'
        )(x)
    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling)([mean, log_var])
    encoder = keras.Model(inputs=encoder_inputs, outputs=[z, mean, log_var])

    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    rev_hidden_layers = reversed(hidden_layers)
    for unit in rev_hidden_layers:
        x = keras.layers.Dense(
            units=unit,
            activation='relu'
        )(x)
    x = keras.layers.Dense(units=input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_inputs, outputs=x)

    z, mean, log_var = encoder(encoder_inputs)
    decoder_outputs = decoder(z)
    auto = keras.Model(inputs=encoder_inputs, outputs=decoder_outputs)

    kl_loss = -0.5 * keras.backend.sum(
        1 + log_var - keras.backend.square(mean) - keras.backend.exp(log_var),
        axis=-1
    )

    auto.add_loss(kl_loss)

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
