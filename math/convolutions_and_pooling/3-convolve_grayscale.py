#!/usr/bin/env python3
"""Module to perform a grayscale convolution on images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Perform a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Array of shape (m, h_orig, w_orig) containing
            the images.
            m is the number of images.
            h_orig is the original height in pixels of the images.
            w_orig is the original width in pixels of the images.
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

    m, h_orig, w_orig = images.shape

    if padding == 'same':
        h_pos = int(np.ceil(h_orig / sh))
        w_pos = int(np.ceil(w_orig / sw))

        ph = max(0, (h_pos - 1) * sh + kh - h_orig)
        pw = max(0, (w_pos - 1) * sw + kw - w_orig)

        bph, bpw = ph // 2, pw // 2
        aph, apw = ph - bph, pw - bpw
        pad = ((0, 0), (bph, aph), (bpw, apw))

    elif padding == 'valid':
        pad = 0
        h_pos = (h_orig - kh) // sh + 1
        w_pos = (w_orig - kw) // sw + 1

    elif isinstance(padding, tuple):
        pad = ((0,), (padding[0],), (padding[1],))
        h_pos = (h_orig + 2 * padding[0] - kh) // sh + 1
        w_pos = (w_orig + 2 * padding[1] - kw) // sw + 1

    images_pad = np.pad(images, pad)

    new_mat = np.zeros((m, h_pos, w_pos))
    total_pos = h_pos * w_pos

    for index in range(total_pos):
        i = index // w_pos
        j = index % w_pos
        part_mat = images_pad[:, i * sh: (i * sh) + kh, j * sw: (j * sw) + kw]

        new_mat[:, i, j] = np.sum(part_mat * kernel, axis=(1, 2))

    return new_mat
