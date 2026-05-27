#!/usr/bin/env python3
"""Module to perform forward propagation over a convolutional layer."""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Perform forward propagation over a convolutional layer of a network.

    Args:
        A_prev (numpy.ndarray): Shape (m, h_prev, w_prev, c_prev)
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W (numpy.ndarray): Shape (kh, kw, c_prev, c_new)
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b (numpy.ndarray): Shape (1, 1, 1, c_new) containing the biases
        activation (function): Activation function applied to the convolution
        padding (str): Either 'same' or 'valid'
        stride (tuple): (sh, sw) strides for height and width

    Returns:
        numpy.ndarray: The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        h_pos = int(np.ceil(h_prev / sh))
        w_pos = int(np.ceil(w_prev / sw))

        ph = max(0, (h_pos - 1) * sh + kh - h_prev) // 2 + 1
        pw = max(0, (w_pos - 1) * sw + kw - w_prev) // 2 + 1

    elif padding == 'valid':
        ph = 0
        pw = 0

    output_h = (h_prev - kh + 2 * ph) // sh + 1
    output_w = (w_prev - kw + 2 * pw) // sw + 1
    output_total = output_h * output_w

    A_prev_padded = np.pad(A_prev, ((0,), (ph,), (pw,), (0,)))

    Z = np.zeros((m, output_h, output_w, c_new))

    for index in range(output_total):
        for k in range(c_new):
            i = index // output_h
            j = index % output_w

            slice_A = A_prev_padded[:, i * sh: i * sh + kh,
                                    j * sw: j * sw + kw, :]

            current_kernel = W[:, :, :, k]

            Z[:, i, j, k] = np.sum(slice_A * current_kernel,
                                   axis=(1, 2, 3)) + b[0, 0, 0, k]

    return activation(Z)
