#!/usr/bin/env python3
"""Module to perform a valid grayscale convolution on images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Perform a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w) containing the images.
            m is the number of images.
            h is the height in pixels of the images.
            w is the width in pixels of the images.
        kernel (numpy.ndarray): Array of shape (kh, kw) containing the kernel
            for the convolution.
            kh is the height of the kernel.
            kw is the width of the kernel.

    Returns:
        numpy.ndarray: An array containing the convolved images of shape
            (m, h - kh + 1, w - kw + 1).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    h_pos = h - kh + 1
    w_pos = w - kw + 1
    total_pos = h_pos * w_pos

    new_mat = np.zeros((m, h_pos, w_pos))

    for index in range(total_pos):
        i = index // w_pos
        j = index % w_pos
        part_mat = images[:, i: i + kh, j: j + kw]

        new_mat[:, i, j] = np.sum(part_mat * kernel, axis=(1, 2))
    return new_mat
