#!/usr/bin/env python3
"""Module that rotates an image by 90 degrees counter-clockwise."""

import tensorflow as tf


def rotate_image(image):
    """Rotate an image counter-clockwise by 90 degrees.

    Args:
        image (tf.Tensor): A 3D tensor of shape (height, width, channels)
            representing the image to rotate.

    Returns:
        tf.Tensor: The rotated image tensor.
    """
    return tf.image.rot90(image)
