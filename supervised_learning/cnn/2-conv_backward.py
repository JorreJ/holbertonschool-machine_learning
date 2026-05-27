#!/usr/bin/env python3
"""Module to perform backward propagation over a convolutional layer."""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Perform backward propagation over a convolutional layer.

    Args:
        dZ (numpy.ndarray): Array of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output.
        A_prev (numpy.ndarray): Array of shape (m, h_prev, w_prev, c_prev)
            containing the activations from the previous layer.
            m is the number of examples.
            h_prev is the height of the previous layer.
            w_prev is the width of the previous layer.
            c_prev is the number of channels in the previous layer.
        W (numpy.ndarray): Array of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution.
            kh is the height of a kernel.
            kw is the width of a kernel.
            c_prev is the number of channels in the previous layer.
            c_new is the number of kernels.
        b (numpy.ndarray): Array of shape (1, 1, 1, c_new) containing
            the biases.
        padding (str or tuple): Can be 'same', 'valid', or a tuple of (ph, pw)
            containing the padding heights and widths respectively.
        stride (tuple): A tuple of (sh, sw) containing the strides for the
            height and width of the convolution respectively.

    Returns:
        tuple: A tuple containing:
            - dA_prev (numpy.ndarray): Gradient of the cost with respect to
              A_prev, of shape (m, h_prev, w_prev, c_prev).
            - dW (numpy.ndarray): Gradient of the cost with respect to W,
              of shape (kh, kw, c_prev, c_new).
            - db (numpy.ndarray): Gradient of the cost with respect to b,
              of shape (1, 1, 1, c_new).
    """
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    m, h_prev, w_prev, c_prev = A_prev.shape
    _, h_new, w_new, _ = dZ.shape

    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    elif padding == 'valid':
        ph = 0
        pw = 0

    A_prev_padded = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)))
    dA_prev_padded = np.zeros_like(A_prev_padded)

    total_pos = h_new * w_new

    for index in range(total_pos):
        for k in range(c_new):
            i = index // w_new
            j = index % w_new
            part_mat = A_prev_padded[:, i * sh: (i * sh) + kh,
                                     j * sw: (j * sw) + kw, :]
            dZ_pixel = dZ[:, i, j, k][:, np.newaxis, np.newaxis, np.newaxis]
            dW[:, :, :, k] += np.sum(part_mat * dZ_pixel, axis=0)
            dA_prev_padded[:, i * sh: (i * sh) + kh, j * sw:
                           (j * sw) + kw, :] += dZ_pixel * W[:, :, :, k]

    if ph > 0 and pw > 0:
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    elif ph > 0:
        dA_prev = dA_prev_padded[:, ph:-ph, :, :]
    elif pw > 0:
        dA_prev = dA_prev_padded[:, :, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
