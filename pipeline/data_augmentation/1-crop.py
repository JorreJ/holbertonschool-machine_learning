#!/usr/bin/env python3
"""Module that performs a random crop on an image using TensorFlow."""

import tensorflow as tf


def crop_image(image, size):
    """Perform a random crop of an image to a specified size.

    Args:
        image (tf.Tensor): A 3D tensor of shape (height, width, channels)
            representing the image to crop.
        size (tuple or list): A 1D tensor or sequence of 3 integers specifying
            the target crop dimensions (crop_height, crop_width, channels).

    Returns:
        tf.Tensor: The randomly cropped image tensor with the specified size.
    """
    return tf.image.random_crop(image, size)
