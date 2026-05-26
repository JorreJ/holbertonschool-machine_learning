#!/usr/bin/env python3
"""Module to perform a grayscale convolution on images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Perform a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w) containing
            the images.
            m is the number of images.
            h is the height in pixels of the images.
            w is the width in pixels of the images.
        kernel (numpy.ndarray): Array of shape (kh, kw) containing the kernel
            for the convolution.
            kh is the height of the kernel.
            kw is the width of the kernel.
        padding (str or tuple): Can be 'same', 'valid', or a tuple of (ph, pw)
            containing the padding heights and widths respectively.
        stride (tuple): A tuple of (sh, sw) containing the strides for the
            height and width of the image respectively.

    Returns:
        numpy.ndarray: An array containing the convolved images.
    """
    kh, kw = kernel.shape
    sh, sw = stride
    m, h, w = images.shape

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1

    elif padding == 'valid':
        ph = 0
        pw = 0

    elif isinstance(padding, tuple):
        ph, pw = padding

    h_pos = (h - kh + 2 * ph) // sh + 1
    w_pos = (w - kw + 2 * pw) // sw + 1

    images_pad = np.pad(images, ((0,), (ph,), (pw,)))

    new_mat = np.zeros((m, h_pos, w_pos))
    total_pos = h_pos * w_pos

    for index in range(total_pos):
        i = index // w_pos
        j = index % w_pos
        part_mat = images_pad[:, i * sh: (i * sh) + kh, j * sw: (j * sw) + kw]

        new_mat[:, i, j] = np.sum(part_mat * kernel, axis=(1, 2))

    return new_mat
