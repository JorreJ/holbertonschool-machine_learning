#!/usr/bin/env python3
"""Module that randomly adjusts the brightness of an image using TensorFlow."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Randomly adjust the brightness of an image by a random factor.

    Args:
        image (tf.Tensor): A 3D tensor of shape (height, width, channels)
            representing the image to adjust.
        max_delta (float): Maximum value for the random delta picked in the
            interval [-max_delta, max_delta). Must be non-negative.

    Returns:
        tf.Tensor: The brightness-adjusted image tensor with the same shape
        and dtype as the input.
    """
    return tf.image.random_brightness(image, max_delta)
