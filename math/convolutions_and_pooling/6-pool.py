#!/usr/bin/env python3
"""Module to perform pooling operations on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform a pooling operation on images.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w, c) containing
            the images.
            m is the number of images.
            h is the height in pixels of the images.
            w is the width in pixels of the images.
            c is the number of channels in the images.
        kernel_shape (tuple): A tuple of (kh, kw) containing the kernel shape.
            kh is the height of the kernel.
            kw is the width of the kernel.
        stride (tuple): A tuple of (sh, sw) containing the strides for the
            height and width of the image respectively.
        mode (str): Indicates the type of pooling, either 'max' or 'avg'.

    Returns:
        numpy.ndarray: An array containing the pooled images of shape
            (m, h_pos, w_pos, c).
    """
    m, h, w, c = np.shape(images)
    kh, kw = kernel_shape
    sh, sw = stride

    h_pos = (h - kh) // sh + 1
    w_pos = (w - kw) // sw + 1

    new_mat = np.zeros((m, h_pos, w_pos, c))
    total_pos = h_pos * w_pos

    for index in range(total_pos):
        i = index // w_pos
        j = index % w_pos
        part_mat = images[:, i * sh: (i * sh) + kh,
                          j * sw: (j * sw) + kw, :]

        if mode == "max":
            new_mat[:, i, j, :] = np.max(part_mat, axis=(1, 2))
        if mode == "avg":
            new_mat[:, i, j, :] = np.mean(part_mat, axis=(1, 2))

    return new_mat
