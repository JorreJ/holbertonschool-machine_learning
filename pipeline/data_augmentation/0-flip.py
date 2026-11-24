#!/usr/bin/env python3
"""Module that flips an image horizontally using TensorFlow."""

import tensorflow as tf


def flip_image(image):
    """Flip an image horizontally (left to right).

    Args:
        image (tf.Tensor): A 3D tensor of shape (height, width, channels)
            representing the image to flip.

    Returns:
        tf.Tensor: The horizontally flipped image tensor with the same shape
        and dtype as the input.
    """
    return tf.image.flip_left_right(image)
