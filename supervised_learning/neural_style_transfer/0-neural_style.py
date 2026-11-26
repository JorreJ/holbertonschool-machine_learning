#!/usr/bin/env python3
"""Module that implements the Neural Style Transfer (NST) class."""

import numpy as np
import tensorflow as tf


class NST:
    """Perform Neural Style Transfer on images.

    Attributes:
        style_layers (list): VGG19 layer names used for style extraction.
        content_layer (str): VGG19 layer name used for content extraction.
        alpha (float): Weight for the content cost.
        beta (float): Weight for the style cost.
        style_image (tf.Tensor): Preprocessed style image tensor.
        content_image (tf.Tensor): Preprocessed content image tensor.
    """

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize an NST instance.

        Args:
            style_image (numpy.ndarray): Image used as a style reference,
                with shape (h, w, 3).
            content_image (numpy.ndarray): Image used as a content reference,
                with shape (h, w, 3).
            alpha (int or float): Weight for content cost. Defaults to 1e4.
            beta (int or float): Weight for style cost. Defaults to 1.

        Raises:
            TypeError: If style_image or content_image is not a 3D
                numpy.ndarray with 3 channels, or if alpha or beta is not a
                non-negative number.
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3
                or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3
                or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError(
                "alpha must be a non-negative number"
            )

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError(
                "beta must be a non-negative number"
            )

        self.alpha = alpha
        self.beta = beta

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

    @staticmethod
    def scale_image(image):
        """Rescale an image so its pixels are in [0, 1] and max side is 512.

        Args:
            image (numpy.ndarray): Input image with shape (h, w, 3).

        Raises:
            TypeError: If image is not a numpy.ndarray with shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image tensor of shape (1, new_h, new_w, 3).
        """
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        scale = 512 / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        image = tf.image.resize(
            image,
            (new_h, new_w),
            method='bicubic'
        )

        image = image / 255.0

        return image
