#!/usr/bin/env python3
"""Module that provides a function to build a vanilla autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create an autoencoder model along with its encoder and decoder models.

    Args:
        input_dims (int): The dimensions of the model input.
        hidden_layers (list): A list containing the number of nodes for each
            hidden layer in the encoder, respectively.
        latent_dims (int): The dimensions of the latent space representation.

    Returns:
        tuple: A tuple (encoder, decoder, auto) where:
            - encoder (keras.Model): The encoder model.
            - decoder (keras.Model): The decoder model.
            - auto (keras.Model): The full autoencoder model.
    """
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for unit in hidden_layers:
        x = keras.layers.Dense(
            units=unit,
            activation='relu'
        )(x)
    x = keras.layers.Dense(units=latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=encoder_inputs, outputs=x)

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

    decoder_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(inputs=encoder_inputs, outputs=decoder_outputs)

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
