#!/usr/bin/env python3
"""Module that provides a function to build a convolutional autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder model, encoder, and decoder.

    Args:
        input_dims (tuple): A tuple with the dimensions of the model input.
        filters (list): A list containing the number of filters for each
            convolutional layer in the encoder, respectively.
        latent_dims (tuple): A tuple containing the dimensions of the latent
            space representation.

    Returns:
        tuple: A tuple (encoder, decoder, auto) where:
            - encoder (keras.Model): The encoder model.
            - decoder (keras.Model): The decoder model.
            - auto (keras.Model): The full convolutional autoencoder model.
    """
    encoder_inputs = keras.Input(shape=(input_dims))
    x = encoder_inputs
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    encoder = keras.Model(inputs=encoder_inputs, outputs=x)

    decoder_inputs = keras.Input(shape=(latent_dims))
    x = decoder_inputs
    rev_filters = list(reversed(filters))
    for i in range(len(rev_filters)):
        if i == len(rev_filters) - 1:
            padding = 'valid'
        else:
            padding = 'same'

        x = keras.layers.Conv2D(
            filters=rev_filters[i],
            kernel_size=(3, 3),
            padding=padding,
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)
    decoder = keras.Model(inputs=decoder_inputs, outputs=x)

    decoder_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(inputs=encoder_inputs, outputs=decoder_outputs)

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto
