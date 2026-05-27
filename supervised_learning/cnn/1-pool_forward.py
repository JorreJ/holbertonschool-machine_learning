#!/usr/bin/env python3
"""Module to perform pooling operations on input activations."""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform a pooling operation on input activations.

    Args:
        A_prev (numpy.ndarray): Array of shape (m, h_prev, w_prev, c_prev)
            containing the activations from the previous layer.
            m is the number of examples.
            h_prev is the height of the previous layer.
            w_prev is the width of the previous layer.
            c_prev is the number of channels in the previous layer.
        kernel_shape (tuple): A tuple of (kh, kw) containing the kernel shape.
            kh is the height of the kernel.
            kw is the width of the kernel.
        stride (tuple): A tuple of (sh, sw) containing the strides for the
            height and width of the pooling operation respectively.
        mode (str): Indicates the type of pooling, either 'max' or 'avg'.

    Returns:
        numpy.ndarray: An array containing the pooled activations of shape
            (m, h_pos, w_pos, c_prev).
    """
    m, h_prev, w_prev, c_prev = np.shape(A_prev)
    kh, kw = kernel_shape
    sh, sw = stride

    h_pos = (h_prev - kh) // sh + 1
    w_pos = (w_prev - kw) // sw + 1

    new_mat = np.zeros((m, h_pos, w_pos, c_prev))
    total_pos = h_pos * w_pos

    for index in range(total_pos):
        i = index // w_pos
        j = index % w_pos
        part_mat = A_prev[:, i * sh: (i * sh) + kh,
                          j * sw: (j * sw) + kw, :]

        if mode == "max":
            new_mat[:, i, j, :] = np.max(part_mat, axis=(1, 2))
        if mode == "avg":
            new_mat[:, i, j, :] = np.mean(part_mat, axis=(1, 2))

    return new_mat
