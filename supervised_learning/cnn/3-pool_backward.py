#!/usr/bin/env python3
"""Module to perform backward propagation over a pooling layer."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform backward propagation over a pooling layer of a neural network.

    Args:
        dA (numpy.ndarray): Array of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the output of the pooling.
        A_prev (numpy.ndarray): Array of shape (m, h_prev, w_prev, c)
            containing the output of the previous layer.
        kernel_shape (tuple): A tuple of (kh, kw) containing the size
            of the kernel for the pooling.
        stride (tuple): A tuple of (sh, sw) containing the strides
            for the pooling.
        mode (str): A string containing either 'max' or 'avg'.

    Returns:
        numpy.ndarray: The partial derivatives with respect to the
        previous layer (dA_prev).
    """
    m, h_new, w_new, c_new = dA.shape
    _, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    total_pos = h_new * w_new

    for index in range(total_pos):
        i = index // w_new
        j = index % w_new

        v_start = i * sh
        v_end = v_start + kh
        h_start = j * sw
        h_end = h_start + kw

        for ex in range(m):
            for ch in range(c):
                dA_pixel = dA[ex, i, j, ch]

                if mode == 'max':
                    slice_A = A_prev[ex, v_start:v_end, h_start:h_end, ch]

                    mask = (slice_A == np.max(slice_A))

                    dA_prev[ex, v_start:v_end,
                            h_start:h_end, ch] += mask * dA_pixel

                elif mode == 'avg':
                    average_gradient = dA_pixel / (kh * kw)

                    dA_prev[ex, v_start:v_end,
                            h_start:h_end, ch] += average_gradient

    return dA_prev
