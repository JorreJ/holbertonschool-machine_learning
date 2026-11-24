#!/usr/bin/env python3
"""Module that randomly adjusts the contrast of an image using TensorFlow."""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjust the contrast of an image within a specified range.

    Args:
        image (tf.Tensor): A 3D tensor of shape (height, width, channels)
            representing the image to adjust.
        lower (float): Lower bound for the random contrast factor.
        upper (float): Upper bound for the random contrast factor.

    Returns:
        tf.Tensor: The contrast-adjusted image tensor with the same shape and
        dtype as the input.
    """
    return tf.image.random_contrast(image, lower, upper)
