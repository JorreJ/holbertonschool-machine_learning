#!/usr/bin/env python3
"""Module that adjusts the hue of an image using TensorFlow."""

import tensorflow as tf


def change_hue(image, delta):
    """Adjust the hue of an RGB image by adding an offset.

    Args:
        image (tf.Tensor): A 3D tensor of shape (height, width, channels)
            representing the RGB image to adjust.
        delta (float): Amount to add to the hue channel. Must be in the
            interval [-1, 1].

    Returns:
        tf.Tensor: The hue-adjusted image tensor with the same shape and
        dtype as the input.
    """
    return tf.image.adjust_hue(image, delta)
