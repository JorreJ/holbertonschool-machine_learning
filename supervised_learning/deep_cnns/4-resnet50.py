#!/usr/bin/env python3
"""Module to construct a ResNet-50 network architecture using Keras."""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Build a ResNet-50 neural network.

    The model expects input images of shape (224, 224, 3) and outputs a
    softmax probability distribution over 1000 classes.

    Returns:
        K.models.Model: The constructed Keras Model instance.
    """
    Input = K.Input(shape=(224, 224, 3))

    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(Input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    X = projection_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, filters=[64, 64, 256])
    X = identity_block(X, filters=[64, 64, 256])

    X = projection_block(X, filters=[128, 128, 512], s=2)
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])

    X = projection_block(X, filters=[256, 256, 1024], s=2)
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])

    X = projection_block(X, filters=[512, 512, 2048], s=2)
    X = identity_block(X, filters=[512, 512, 2048])
    X = identity_block(X, filters=[512, 512, 2048])

    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)
    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(X)

    model = K.models.Model(inputs=Input, outputs=X, name='model')

    return model
